#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace fs = std::filesystem;
using Text = std::string_view;
using Id = uint64_t;
using Time = int64_t;
constexpr Time day_ms = 86'400'000, reach = 7 * day_ms;
constexpr Time bad_time = std::numeric_limits<Time>::min();
constexpr size_t partitions = 64;
constexpr Id extended = Id{1} << 63;

uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct IdHash {
    size_t operator()(Id id) const { return mix(id); }
};

class Ids {
    std::mutex mutex;
    std::unordered_map<Text, Id> extra;
    std::vector<Text> strings;
public:
    Id encode(Text s) {
        #ifdef __BMI2__
        if (s.size() == 8 || s.size() == 9) {
            uint64_t bytes;
            memcpy(&bytes, s.data(), sizeof(bytes));
            constexpr uint64_t low = 0x0101010101010101ULL;
            constexpr uint64_t high = 0x8080808080808080ULL;
            if (!(bytes & high) && !((bytes - low) & ~bytes & high) &&
                (s.size() == 8 || (s[8] > 0 && unsigned(s[8]) < 128))) {
                auto prefix = _pext_u64(__builtin_bswap64(bytes), ~high) << 7;
                return prefix | (s.size() == 9 ? s[8] : 0);
            }
        }
#endif
        if (!s.empty() && s.size() <= 9) {
            Id key = 0;
            for (unsigned char ch : s) {
                if (ch == 0 || ch > 127) { key = 0; break; }
                key = (key << 7) | ch;
            }
            if (key) return key << (7 * (9 - s.size()));
        }
        std::lock_guard lock(mutex);
        auto [it, added] = extra.try_emplace(s, extended | strings.size());
        if (added) strings.push_back(s);
        return it->second;
    }

    std::string decode(Id id) const {
        if (id & extended) return std::string(strings[id ^ extended]);
        std::string s;
        for (int shift = 56; shift >= 0 && ((id >> shift) & 127); shift -= 7)
            s.push_back(char((id >> shift) & 127));
        return s;
    }

    bool less(Id a, Id b) const {
        if ((a | b) < extended) return a < b;
        return decode(a) < decode(b);
    }
};

struct Mapping {
    const char* data = nullptr;
    size_t size = 0;
    explicit Mapping(const fs::path& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open " + path.string());
        struct stat info {};
        if (fstat(fd, &info) != 0) {
            close(fd);
            throw std::runtime_error("Cannot stat " + path.string());
        }
        size = info.st_size;
        if (size) data = static_cast<const char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);
        if (data == MAP_FAILED) throw std::runtime_error("Cannot map " + path.string());
    }
    Mapping(Mapping&& other) noexcept : data(other.data), size(other.size) { other.size = 0; }
    ~Mapping() { if (size) munmap(const_cast<char*>(data), size); }
};

const char* line_end(const char* begin, const char* end) {
    auto *nl = static_cast<const char*>(memchr(begin, '\n', end - begin));
    return nl ? nl : end;
}

template<size_t N>
bool fields(const char* begin, const char* end, const char* limit, std::array<Text,N>& row) {
    std::array<const char*, N> cuts;
    size_t count = 0;
    const char* at = begin;
#ifdef __AVX2__
    size_t quotes_seen = 0;
    for (; at < end && limit - at >= 32; at += 32) {
        auto block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(at));
        uint32_t commas = _mm256_movemask_epi8(_mm256_cmpeq_epi8(block, _mm256_set1_epi8(',')));
        uint32_t quotes = _mm256_movemask_epi8(_mm256_cmpeq_epi8(block, _mm256_set1_epi8('"')));
        if (end - at < 32) {
            auto mask = (uint32_t{1} << (end - at)) - 1;
            commas &= mask;
            quotes &= mask;
        }
        quotes_seen += std::popcount(quotes);
        if (count + std::popcount(commas) >= N) return false;
        while (commas) {
            cuts[count++] = at + std::countr_zero(commas);
            commas &= commas - 1;
        }
    }
#else
    size_t quotes_seen = 0;
    (void)limit;
#endif
    for (; at < end; ++at) {
        if (*at == ',') {
            if (count == N - 1) return false;
            cuts[count++] = at;
        }
        quotes_seen += *at == '"';
    }
    if (count != N - 1) return false;
    cuts[count] = end;
    for (size_t i = 0; i < N; ++i) {
        auto length = cuts[i] - begin;
        auto value = begin;
        if (length >= 2 && value[0] == '"' && value[length - 1] == '"') {
            ++value;
            length -= 2;
            quotes_seen -= 2;
        }
        row[i] = Text(value, length);
        begin = cuts[i] + (i + 1 != N);
    }
    return quotes_seen == 0;
}

bool digit(char ch) { return unsigned(ch - '0') <= 9; }

Time decimal(Text s, Time maximum) {
    if (s.empty()) return bad_time;
    Time value = 0;
    // Up to 18 decimal characters fit in int64 without per-digit overflow checks.
    if (maximum == std::numeric_limits<Time>::max() && s.size() <= 18) {
        for (char ch : s) {
            if (!digit(ch)) return bad_time;
            value = value * 10 + ch - '0';
        }
        return value;
    }
    for (char ch : s) {
        if (!digit(ch) || value > (maximum - (ch - '0')) / 10) return bad_time;
        value = 10 * value + ch - '0';
    }
    return value;
}

int pair_at(const char* s) {
    if (!digit(s[0]) || !digit(s[1])) return -1;
    return 10 * (s[0] - '0') + s[1] - '0';
}

struct Calendar {
    // One table handles calendar validation and the hot date-to-epoch conversion.
    std::array<int32_t, 10000 * 13> month_start{};
    Calendar() {
        using namespace std::chrono;
        for (int y = 0; y < 10000; ++y)
            for (unsigned m = 1; m <= 12; ++m)
                month_start[y * 13 + m - 1] = sys_days{year{y}/m/1}.time_since_epoch().count();
        for (int y = 0; y < 10000; ++y)
            month_start[y * 13 + 12] = month_start[y * 13 + 11] + 31;
    }

    Time iso(Text s) const {
        const size_t n = s.size();
        if (n != 20 && n != 25) return bad_time;
        const char* p = s.data();
        if (p[4] != '-' || p[7] != '-' || p[10] != 'T' || p[13] != ':' || p[16] != ':') return bad_time;
        int century = pair_at(p), year = pair_at(p+2), month = pair_at(p+5), day = pair_at(p+8);
        int hour = pair_at(p+11), minute = pair_at(p+14), second = pair_at(p+17);
        if (century < 0 || year < 0 || month < 1 || month > 12 || day < 1 ||
            unsigned(hour) >= 24 || unsigned(minute) >= 60 || unsigned(second) >= 60)
            return bad_time;
        int idx = (century * 100 + year) * 13 + month - 1;
        if (day > month_start[idx+1] - month_start[idx]) return bad_time;
        int zone = 0;
        if (n == 20) {
            if (p[19] != 'Z') return bad_time;
        } else {
            int zh = pair_at(p+20), zm = pair_at(p+23);
            if ((p[19] != '+' && p[19] != '-') || p[22] != ':' || unsigned(zh) >= 24 || unsigned(zm) >= 60)
                return bad_time;
            zone = (zh * 60 + zm) * (p[19] == '+' ? 1 : -1);
        }
        return ((Time(month_start[idx] + day - 1) * 24 + hour) * 3600 +
                (minute - zone) * 60 + second) * 1000;
    }

    Time ingest(Text s) const {
        // Input contract guarantees F1 for ingestion timestamps.
        const char* p = s.data();
        auto two = [](const char* q) { return 10 * q[0] + q[1] - 11 * '0'; };
        int year = two(p) * 100 + two(p + 2);
        int index = year * 13 + two(p + 5) - 1;
        if (unsigned(index) >= month_start.size()) return bad_time;
        int day = month_start[index] + two(p + 8) - 1;
        return Time(day) * day_ms + two(p + 11) * 3'600'000 +
               two(p + 14) * 60'000 + two(p + 17) * 1000;
    }

    Time timestamp(Text s) const {
        if (s.size() > 4 && s[4] == '-') return iso(s);
        return decimal(s, std::numeric_limits<Time>::max());
    }
};

struct Chunk { const char *begin, *end, *limit; };
void add_chunks(const Mapping& map, std::vector<Chunk>& chunks) {
    if (!map.size) return;
    const char* limit = map.data + map.size;
    const char* header = line_end(map.data, limit);
    if (header == limit) return;
    for (const char* begin = header + 1; begin < limit;) {
        const char* end = limit;
        constexpr ptrdiff_t target = 2 << 20;
        if (limit - begin > target) {
            end = line_end(begin + target, limit);
            if (end < limit) ++end;
        }
        chunks.push_back({begin, end, limit});
        begin = end;
    }
}

template<size_t N, typename F>
void read_rows(Chunk chunk, F consume) {
#if defined(__AVX512BW__)
    // Scan each 64-byte block once. The three masks describe the CSV syntax
    // independently of the row and field data carried across the block boundary.
    const char* field = chunk.begin;
    std::array<Text,N> row;
    size_t count = 0;
    size_t quote_count = 0;
    bool valid = true;
    for (const char* block = chunk.begin; block < chunk.end;) {
        const auto remaining = chunk.end - block;
        const auto active = remaining >= 64 ? ~uint64_t{0} : (uint64_t{1} << remaining) - 1;
        const auto input = remaining >= 64 ? _mm512_loadu_si512(block) :
                                             _mm512_maskz_loadu_epi8(active, block);
        auto commas = _mm512_cmpeq_epi8_mask(input, _mm512_set1_epi8(','));
        auto lines = _mm512_cmpeq_epi8_mask(input, _mm512_set1_epi8('\n'));
        auto quotes = _mm512_cmpeq_epi8_mask(input, _mm512_set1_epi8('"'));
        auto boundaries = commas | lines;
        while (boundaries) {
            unsigned at = std::countr_zero(boundaries);
            uint64_t through = boundaries ^ (boundaries - 1);
            quote_count += std::popcount(quotes & through);
            quotes &= ~through;
            boundaries &= boundaries - 1;
            const char* cut = block + at;
            if (count < N) {
                bool wrapped = cut - field >= 2 && field[0] == '"' && cut[-1] == '"';
                row[count++] = Text(field + wrapped, cut - field - 2 * wrapped);
                valid &= quote_count == 2 * size_t(wrapped);
            } else {
                valid = false;
            }
            field = cut + 1;
            quote_count = 0;
            if ((lines >> at) & 1) {
                if (valid && count == N) consume(row);
                count = 0;
                valid = true;
            }
        }
        quote_count += std::popcount(quotes);
        block += std::min<ptrdiff_t>(remaining, 64);
    }
    if (field != chunk.end || count) {
        if (valid && count == N - 1) {
            auto cut = chunk.end;
            bool wrapped = cut - field >= 2 && field[0] == '"' && cut[-1] == '"';
            if (quote_count == 2 * size_t(wrapped)) {
                row[count] = Text(field + wrapped, cut - field - 2 * wrapped);
                consume(row);
            }
        }
    }
#else
    const char* begin = chunk.begin;
    while (begin < chunk.end) {
        const char* end = line_end(begin, chunk.end);
        std::array<Text,N> row;
        if (fields(begin, end, chunk.limit, row)) consume(row);
        if (end == chunk.end) break;
        begin = end + 1;
    }
#endif
}

template<typename F>
void parallel(size_t jobs, unsigned workers, F work) {
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex error_lock;
    auto run = [&](unsigned worker) {
        try {
            for (size_t job; (job = next.fetch_add(1, std::memory_order_relaxed)) < jobs;)
                work(job, worker);
        } catch (...) {
            std::lock_guard lock(error_lock);
            if (!error) error = std::current_exception();
            next.store(jobs, std::memory_order_relaxed);
        }
    };
    std::vector<std::jthread> threads;
    for (unsigned i = 1; i < std::min<size_t>(jobs, workers); ++i)
        threads.emplace_back(run, i);
    run(0);
    threads.clear();
    if (error) std::rethrow_exception(error);
}

struct Links {
    std::unordered_map<Id, uint32_t, IdHash> index;
    std::vector<uint32_t> parent, size;
    std::vector<Id> names;
    struct Alias { Id id{}, name{}; };
    std::vector<Alias> lookup;
    uint32_t add(Id id) {
        auto [it, inserted] = index.try_emplace(id, parent.size());
        if (inserted) {
            parent.push_back(it->second);
            size.push_back(1);
            names.push_back(id);
        }
        return it->second;
    }
    uint32_t root(uint32_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }
    void join(Id a, Id b) {
        uint32_t left = root(add(a)), right = root(add(b));
        if (left == right) return;
        if (size[left] < size[right]) std::swap(left, right);
        parent[right] = left;
        size[left] += size[right];
    }
    void finish(const Ids& ids) {
        for (size_t i = 0; i < parent.size(); ++i) {
            auto r = root(i);
            parent[i] = r;
            if (ids.less(names[i], names[r])) names[r] = names[i];
        }
        lookup.resize(std::bit_ceil(index.size() * 2 + 1));
        for (auto [id, slot] : index) {
            size_t pos = mix(id) & (lookup.size() - 1);
            while (lookup[pos].id) pos = (pos + 1) & (lookup.size() - 1);
            lookup[pos] = {id, names[parent[slot]]};
        }
        decltype(index)().swap(index);
        decltype(parent)().swap(parent);
        decltype(size)().swap(size);
        decltype(names)().swap(names);
    }
    Id person(Id id) const {
        size_t pos = mix(id) & (lookup.size() - 1);
        while (lookup[pos].id) {
            if (lookup[pos].id == id) return lookup[pos].name;
            pos = (pos + 1) & (lookup.size() - 1);
        }
        return id;
    }
};

struct Catalogs {
    std::unordered_map<Text, uint32_t> product, campaign;
    std::vector<Text> categories{"unknown"}, campaigns{"organic"}, channels{"organic"};
};

struct Candidate { Id id, person; Time ts, ingest; uint32_t price, category; };
struct Event { Id person; Time ts; uint32_t price, category; };
struct Click { Id person, id; Time ts; uint32_t campaign; };
struct Session { Time start; int64_t revenue; };
struct UserDay { Id person; int32_t day; int64_t sessions, events, revenue; int32_t top; };
struct CampaignDay { int64_t sessions{}, purchasing{}, revenue{}, assists{}; };
using CampaignTotals = std::unordered_map<uint64_t, CampaignDay, IdHash>;

template<typename T> using Buckets = std::array<std::vector<T>, partitions>;
size_t bucket(Id id) { return mix(id) & (partitions - 1); }

int32_t utc_day(Time ts) {
    return static_cast<int32_t>(ts / day_ms - (ts % day_ms < 0));
}

uint64_t campaign_day(uint32_t campaign, int32_t day) {
    return (uint64_t(campaign) << 32) | uint32_t(day);
}

struct DayRange { int32_t day; size_t begin, end; };

void attribute(const Ids& ids, const std::vector<Session>& sessions, const std::vector<DayRange>& days,
               const Click* clicks, size_t count, CampaignTotals& totals) {
    size_t next = 0;
    const Click* latest = nullptr;
    for (const auto& session : sessions) {
        while (next < count && clicks[next].ts <= session.start) {
            auto& c = clicks[next++];
            if (!latest || c.ts > latest->ts || (c.ts == latest->ts && ids.less(c.id, latest->id)))
                latest = &c;
        }
        auto campaign = latest && latest->ts >= session.start - reach ? latest->campaign : 0;
        auto& row = totals[campaign_day(campaign, utc_day(session.start))];
        ++row.sessions;
        row.purchasing += session.revenue > 0;
        row.revenue += session.revenue;
    }

    auto add_interval = [&](uint32_t campaign, Time first, Time last) {
        for (const auto& day : days) {
            const auto* begin = sessions.data() + day.begin;
            const auto* end = sessions.data() + day.end;
            if (last < begin->start || first > (end-1)->start) continue;
            auto l = std::lower_bound(begin, end, first, [](const Session& s, Time t){ return s.start < t; });
            auto r = std::upper_bound(l, end, last, [](Time t, const Session& s){ return t < s.start; });
            if (l != r) totals[campaign_day(campaign, day.day)].assists += r - l;
        }
    };

    // Coalesce overlapping windows per campaign before counting sessions in them.
    // Identical duplicate clicks have no effect on either credit or these unions.
    struct Interval { Time first, last; };
    if (count == 1) add_interval(clicks->campaign, clicks->ts, clicks->ts + reach);
    if (count <= 1) return;
    std::unordered_map<uint32_t, Interval> windows;
    for (size_t i = 0; i < count && clicks[i].ts <= sessions.back().start; ++i) {
        auto& c = clicks[i];
        if (c.ts < sessions.front().start - reach) continue;
        auto [it, inserted] = windows.try_emplace(c.campaign, Interval{c.ts, c.ts + reach});
        if (inserted) continue;
        auto& interval = it->second;
        if (c.ts > interval.last) {
            add_interval(c.campaign, interval.first, interval.last);
            interval.first = c.ts;
        }
        interval.last = c.ts + reach;
    }
    for (auto [campaign, interval] : windows) add_interval(campaign, interval.first, interval.last);
}

void aggregate(std::vector<Event>& events, std::vector<Click>& clicks, const Catalogs& catalogs,
               const Ids& ids, std::vector<UserDay>& users, CampaignTotals& totals) {
    auto by_person_time = [](const auto& a, const auto& b) {
        return a.person != b.person ? a.person < b.person : a.ts < b.ts;
    };
    std::sort(events.begin(), events.end(), by_person_time);
    std::sort(clicks.begin(), clicks.end(), by_person_time);
    std::vector<Session> sessions;
    std::vector<DayRange> days;
    std::vector<int64_t> category_totals(catalogs.categories.size());
    std::vector<uint32_t> touched;
    size_t click = 0;
    for (size_t i = 0; i < events.size();) {
        const Id person = events[i].person;
        sessions.clear();
        days.clear();
        Time previous = events[i].ts;
        auto finish_day = [&] {
            auto& row = users.back();
            for (uint32_t cat : touched) {
                if (row.top < 0 || category_totals[cat] > category_totals[row.top] ||
                    (category_totals[cat] == category_totals[row.top] && catalogs.categories[cat] < catalogs.categories[row.top]))
                    row.top = cat;
            }
            for (uint32_t cat : touched) category_totals[cat] = 0;
            touched.clear();
            days.back().end = sessions.size();
        };
        do {
            const auto& event = events[i++];
            if (sessions.empty() || event.ts - previous > 1'800'000) {
                auto day = utc_day(event.ts);
                if (days.empty() || days.back().day != day) {
                    if (!days.empty()) finish_day();
                    days.push_back({day, sessions.size(), 0});
                    users.push_back({person, day, 0, 0, 0, -1});
                }
                sessions.push_back({event.ts, 0});
                ++users.back().sessions;
            }
            previous = event.ts;
            ++users.back().events;
            if (event.price) {
                users.back().revenue += event.price;
                sessions.back().revenue += event.price;
                auto& total = category_totals[event.category];
                if (!total) touched.push_back(event.category);
                total += event.price;
            }
        } while (i < events.size() && events[i].person == person);
        finish_day();

        while (click < clicks.size() && clicks[click].person < person) ++click;
        size_t first = click;
        while (click < clicks.size() && clicks[click].person == person) ++click;
        attribute(ids, sessions, days, first == click ? nullptr : clicks.data() + first, click - first, totals);
    }
}

template<typename T>
void write_values(std::ofstream& out, const std::vector<T>& values) {
    if (!values.empty()) out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}

template<typename T, typename Rows, typename F>
void write_column(std::ofstream& out, const Rows& rows, F value) {
    std::vector<T> values;
    values.reserve(rows.size());
    for (const auto& row : rows) values.push_back(value(row));
    write_values(out, values);
}

template<typename Rows, typename F>
void write_strings(std::ofstream& out, const Rows& rows, F value) {
    std::vector<int32_t> offsets{0};
    std::string data;
    offsets.reserve(rows.size() + 1);
    for (const auto& row : rows) {
        auto s = value(row);
        if (data.size() + s.size() > INT32_MAX) throw std::runtime_error("String column exceeds 2 GiB");
        data.append(s);
        offsets.push_back(data.size());
    }
    write_values(out, offsets);
    out.write(data.data(), data.size());
}

std::ofstream binary_table(const fs::path& path, uint64_t rows) {
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    return out;
}

void write_users(const fs::path& path, const std::vector<UserDay>& rows,
                 const Ids& ids, const Catalogs& catalogs) {
    auto out = binary_table(path, rows.size());
    write_strings(out, rows, [&](const auto& r) { return ids.decode(r.person); });
    write_column<int32_t>(out, rows, [](const auto& r) { return r.day; });
    write_column<int64_t>(out, rows, [](const auto& r) { return r.sessions; });
    write_column<int64_t>(out, rows, [](const auto& r) { return r.events; });
    write_column<int64_t>(out, rows, [](const auto& r) { return r.revenue; });
    std::vector<uint8_t> validity((rows.size() + 7) / 8);
    for (size_t i = 0; i < rows.size(); ++i)
        if (rows[i].top >= 0) validity[i / 8] |= 1 << (i % 8);
    write_values(out, validity);
    write_strings(out, rows, [&](const auto& r) { return r.top < 0 ? Text{} : catalogs.categories[r.top]; });
    out.close();
}

void write_campaigns(const fs::path& path, const std::vector<CampaignTotals>& shards,
                     const Catalogs& catalogs) {
    CampaignTotals rows;
    for (const auto& shard : shards)
        for (const auto& [key, value] : shard) {
            auto& row = rows[key];
            row.sessions += value.sessions;
            row.purchasing += value.purchasing;
            row.revenue += value.revenue;
            row.assists += value.assists;
        }
    auto out = binary_table(path, rows.size());
    write_strings(out, rows, [&](const auto& r) { return catalogs.campaigns[r.first >> 32]; });
    write_strings(out, rows, [&](const auto& r) { return catalogs.channels[r.first >> 32]; });
    write_column<int32_t>(out, rows, [](const auto& r) { return std::bit_cast<int32_t>(uint32_t(r.first)); });
    write_column<int64_t>(out, rows, [](const auto& r) { return r.second.sessions; });
    write_column<int64_t>(out, rows, [](const auto& r) { return r.second.purchasing; });
    write_column<int64_t>(out, rows, [](const auto& r) { return r.second.revenue; });
    write_column<int64_t>(out, rows, [](const auto& r) { return r.second.assists; });
    out.close();
}

template<typename T>
std::vector<T> combine(std::vector<Buckets<T>>& local, size_t shard) {
    std::vector<T> combined;
    size_t count = 0;
    for (auto& worker : local) count += worker[shard].size();
    combined.reserve(count);
    for (auto& worker : local) {
        auto& source = worker[shard];
        combined.insert(combined.end(), source.begin(), source.end());
        std::vector<T>().swap(source);
    }
    return combined;
}

int run(const fs::path& data, const fs::path& output, const Calendar& calendar,
        Time start, Time end, unsigned workers) {
    Ids ids;
    Links links;
    Catalogs catalogs;
    std::vector<Mapping> mappings;
    auto load = [&](const fs::path& file) {
        std::vector<Chunk> chunks;
        mappings.emplace_back(file);
        add_chunks(mappings.back(), chunks);
        return chunks;
    };
    auto folder = [&](Text name) {
        std::vector<Chunk> chunks;
        for (const auto& file : fs::directory_iterator(data / name))
            if (file.path().extension() == ".csv" && file.is_regular_file()) {
                mappings.emplace_back(file.path());
                add_chunks(mappings.back(), chunks);
            }
        return chunks;
    };
    std::unordered_map<Text, uint32_t> category;
    for (auto chunk : load(data / "products.csv"))
        read_rows<2>(chunk, [&](auto& r) {
            auto [it, added] = category.try_emplace(r[1], catalogs.categories.size());
            if (added) catalogs.categories.push_back(r[1]);
            catalogs.product.emplace(r[0], it->second);
        });
    for (auto chunk : load(data / "campaigns.csv"))
        read_rows<2>(chunk, [&](auto& r) {
            catalogs.campaign.emplace(r[0], catalogs.campaigns.size());
            catalogs.campaigns.push_back(r[0]);
            catalogs.channels.push_back(r[1]);
        });
    for (auto chunk : folder("links"))
        read_rows<2>(chunk, [&](auto& r) {
            if (!r[0].empty() && !r[1].empty()) links.join(ids.encode(r[0]), ids.encode(r[1]));
        });
    links.finish(ids);

    auto event_chunks = folder("events");
    auto click_chunks = folder("clicks");
    std::vector<Buckets<Candidate>> candidates(workers);
    std::vector<Buckets<Click>> clicks(workers);
    auto phase = std::chrono::steady_clock::now();
    auto timing = [&](const char* name) {
        auto now = std::chrono::steady_clock::now();
        if (std::getenv("SHOPPING_TIMINGS"))
            std::cerr << name << ' ' << std::chrono::duration<double>(now - phase).count() << " s\n";
        phase = now;
    };
    parallel(event_chunks.size() + click_chunks.size(), workers, [&](size_t job, unsigned worker) {
        if (job < event_chunks.size()) {
            read_rows<7>(event_chunks[job], [&](auto& r) {
                if (r[0].empty() || r[3].size() != 20) return;
                Id id = ids.encode(r[0]);
                Time ingest = calendar.ingest(r[3]);
                Time ts = bad_time;
                uint32_t price = 0, cat = 0;
                Id person = 0;
                if (!r[1].empty() && !r[5].empty() && (r[4] == "view" || r[4] == "cart" || r[4] == "purchase")) {
                    auto parsed = calendar.timestamp(r[2]);
                    if (ingest != bad_time && parsed >= start && parsed < end && ingest - parsed <= day_ms) {
                        if (r[4] == "purchase") {
                            auto cents = decimal(r[6], 10'000'000);
                            if (cents > 0) {
                                price = cents;
                                auto found = catalogs.product.find(r[5]);
                                if (found != catalogs.product.end()) cat = found->second;
                            }
                        }
                        if (r[4] != "purchase" || price) {
                            ts = parsed;
                            person = links.person(ids.encode(r[1]));
                        }
                    }
                }
                // Retain invalid candidates so an earlier invalid copy masks all later copies.
                candidates[worker][bucket(id)].push_back({id, person, ts, ingest, price, cat});
            });
        } else {
            read_rows<4>(click_chunks[job - event_chunks.size()], [&](auto& r) {
                if (r[0].empty() || r[1].empty()) return;
                auto ts = calendar.timestamp(r[3]);
                if (ts < start - reach || ts >= end) return;
                auto campaign = catalogs.campaign.find(r[2]);
                if (campaign == catalogs.campaign.end()) return;
                Id person = links.person(ids.encode(r[1]));
                clicks[worker][bucket(person)].push_back({person, ids.encode(r[0]), ts, campaign->second});
            });
        }
    });
    timing("parse");
    std::vector<Buckets<Event>> events(workers);
    parallel(partitions, workers, [&](size_t shard, unsigned worker) {
        size_t count = 0;
        for (auto& input : candidates) count += input[shard].size();
        if (!count) return;
        // The low hash bits already select the partition. Use the remaining bits for probing.
        size_t capacity = std::bit_ceil(count + count / 2 + 1);
        std::vector<const Candidate*> winners(capacity);
        for (auto& input : candidates) {
            for (const auto& candidate : input[shard]) {
                size_t pos = (mix(candidate.id) / partitions) & (capacity - 1);
                while (winners[pos] && winners[pos]->id != candidate.id)
                    pos = (pos + 1) & (capacity - 1);
                if (!winners[pos] || candidate.ingest < winners[pos]->ingest) winners[pos] = &candidate;
            }
        }
        for (const auto* winner : winners) {
            if (winner && winner->ts != bad_time)
                events[worker][bucket(winner->person)].push_back(
                    {winner->person, winner->ts, winner->price, winner->category});
        }
        for (auto& input : candidates) std::vector<Candidate>().swap(input[shard]);
    });
    timing("dedup");
    std::vector<CampaignTotals> campaigns(partitions);
    parallel(partitions, workers, [&](size_t shard, unsigned) {
        auto shard_events = combine(events, shard);
        auto shard_clicks = combine(clicks, shard);
        std::vector<UserDay> users;
        aggregate(shard_events, shard_clicks, catalogs, ids, users, campaigns[shard]);
        if (!users.empty() || shard == 0)
            write_users(output / ("users-" + std::to_string(shard)), users, ids, catalogs);
    });
    write_campaigns(output / "campaigns", campaigns, catalogs);
    timing("aggregate and encode");
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("Expected data directory, staging directory, window start, window end, and thread count");
        Calendar calendar;
        auto start = calendar.iso(argv[3]), end = calendar.iso(argv[4]);
        if (start == bad_time || end == bad_time || end <= start)
            throw std::runtime_error("Invalid manifest window");
        return run(argv[1], argv[2], calendar, start, end, std::clamp(std::stoi(argv[5]), 1, 64));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
