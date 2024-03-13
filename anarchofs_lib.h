/// Proposal for Non-Uniform Storage Access (NUSA): DEMocracy ON IO (DEMONIO)
/// The following describes a protocol to unionize several filesystems. In general, files and
/// directories on any of the unionized filesystems will be visible by the other filesystems. And
/// when the same file path exists in several filesystems, all filesystem will see the concatenation
/// of all files in a consistent order.
///
/// # Read-only description
///
/// Possible operations:
///
/// - file read:
///   the subsequent read operations are as if the existent files were concatenated into one
///
/// - directory list:
///   aggregate all entries for all filesystems
///   if there is a path that is a file in one filesystem and a directory in other filesystem,
///   represented path as a file
///
/// # Read/write description: [not supported]
///
/// Possible states:
///
/// - [0]: nonexistent or no operation going on among all filesystems
/// - [lo]: opened for local operation at some filesystem
/// - [go]: opened for global operations at some filesystem
///
/// Possible operations:
///
/// - file creation/truncation+read+write:
///   wait until the file has no [go] state on all filesystems
///   set the state to [lo]
///   the subsequent read/write operations are local to the filesystem
///   the closing operation sets the local state to [0]
///
/// - file read+write:
///   wait until the file has no [lo] state on all filesystems
///   set the state to [go]
///   the subsequent read/write operations are as if the existent files were concatenated into one
///   the closing operation sets the local state to [0]
///
/// - directory creation:
///   local operation to the filesystem
///
/// - directory list:
///   aggregate all entries for all filesystems
///   if there is a path that is a file in one filesystem and a directory in other filesystem,
///   represented path as a file

#ifdef AFS_DAEMON_USE_FUSE
#    define FUSE_USE_VERSION 31
#    include <fuse.h>
#endif

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace anarchofs {

    enum class FileType : char { NotExists = 0, RegularFile, Link, Directory, Other };

    using Offset = std::size_t;

    struct FilenameType {
        std::string filename;
        FileType type;
    };

    using FileId = unsigned int;
    constexpr FileId no_file_id = 0;

    /// Common functions to daemons, server and client

    namespace detail {
        /// Write the given object into a string
        /// \param t: object to write
        /// \param s: pointer to the first element to write into

        template <typename T> void write_as_chars(const T &t, char *s) {
            std::copy_n((char *)&t, sizeof(T), s);
        }

        /// Read an object of type `T` from a string
        /// \param s: pointer to the first element to read

        template <typename T> T read_from_chars(const char *s) {
            T t;
            std::copy_n(s, sizeof(T), (char *)&t);
            return t;
        }

        inline unsigned int &get_proc_id();

#ifdef ANARCOFS_LOG
        template <typename... Args> void log(const char *s, Args... args) {
#    ifdef AFS_DAEMON_USE_FUSE
            fuse_log(FUSE_LOG_DEBUG, s, args...);
#    else
            printf("[%d] ", get_proc_id());
            printf(s, args...);
            fflush(stdout);
#    endif
        }

        template <typename... Args> void warning(const char *s, Args... args) {
#    ifdef AFS_DAEMON_USE_FUSE
            fuse_log(FUSE_LOG_DEBUG, s, args...);
#    else
            printf("[%d] warning: ", get_proc_id());
            printf(s, args...);
            fflush(stdout);
#    endif
        }
#else
        template <typename... Args> void log(const char *, Args...) {}
        template <typename... Args> void warning(const char *, Args...) {}
#endif

        template <typename... Args> void show_error(const char *s, Args... args) {
#ifdef AFS_DAEMON_USE_FUSE
            fuse_log(FUSE_LOG_DEBUG, s, args...);
#else
            printf("[%d] error: ", get_proc_id());
            printf(s, args...);
            fflush(stdout);
#endif
        }

        inline void check_mpi(int error) {
            if (error == MPI_SUCCESS) return;

            char s[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(error, s, &len);

#    define CHECK_AND_THROW(ERR)                                                                   \
        if (error == ERR) {                                                                        \
            std::stringstream ss;                                                                  \
            ss << "MPI error: " #ERR ": " << std::string(&s[0], &s[0] + len);                      \
            throw std::runtime_error(ss.str());                                                    \
        }

            CHECK_AND_THROW(MPI_ERR_BUFFER);
            CHECK_AND_THROW(MPI_ERR_COUNT);
            CHECK_AND_THROW(MPI_ERR_TYPE);
            CHECK_AND_THROW(MPI_ERR_TAG);
            CHECK_AND_THROW(MPI_ERR_COMM);
            CHECK_AND_THROW(MPI_ERR_RANK);
            CHECK_AND_THROW(MPI_ERR_ROOT);
            CHECK_AND_THROW(MPI_ERR_GROUP);
            CHECK_AND_THROW(MPI_ERR_OP);
            CHECK_AND_THROW(MPI_ERR_TOPOLOGY);
            CHECK_AND_THROW(MPI_ERR_DIMS);
            CHECK_AND_THROW(MPI_ERR_ARG);
            CHECK_AND_THROW(MPI_ERR_UNKNOWN);
            CHECK_AND_THROW(MPI_ERR_TRUNCATE);
            CHECK_AND_THROW(MPI_ERR_OTHER);
            CHECK_AND_THROW(MPI_ERR_INTERN);
            CHECK_AND_THROW(MPI_ERR_IN_STATUS);
            CHECK_AND_THROW(MPI_ERR_PENDING);
            CHECK_AND_THROW(MPI_ERR_REQUEST);
            CHECK_AND_THROW(MPI_ERR_LASTCODE);
#    undef CHECK_AND_THROW
        }

        /// Client actions

        enum class ClientAction : int {
            GlobalOpenRequest,
            /// Package description:
            /// - request_action:int = GlobalOpenRequest (tag)
            /// - path:char[*]
            ///
            /// Answer:
            /// - file_id::size_t (> 0, or == 0 for error)

            ReadRequest,
            /// Package description:
            /// - request_action:int = ReadRequest (tag)
            /// - file_id:uint32
            /// - offset:size_t
            /// - size:size_t
            ///
            /// Answer if success:
            /// - request_action:int = ReadRequest (tag)
            /// - content:char[*]
            /// Answer if error:
            /// - request_action:int = -1 (tag)
            /// - error_code:int64_t

            CloseRequest,
            /// Package description:
            /// - request_action:int = CloseRequest (tag)
            /// - file_id:uint32
            ///
            /// Answer:
            /// - status:uint32 (== 0 for ok, otherwise for error)
        };
    }

#ifdef BUILD_AFS_DAEMON
    namespace detail {

        /// Performance metrics, time, memory usage, etc

        struct Metric {
            double cpu_time;   ///< wall-clock time for the cpu
            double memops;     ///< bytes read and write from memory
            std::size_t calls; ///< number of times the function was called
            Metric() : cpu_time(0), memops(0), calls(0) {}
        };

        /// Type for storing the timings

        using Timings = std::unordered_map<std::string, Metric>;

        /// Return the performance timings

        inline Timings &getTimings() {
            static Timings timings{16};
            return timings;
        }

        /// Track time between creation and destruction of the object

        struct tracker {
            /// Bytes read and write from memory
            double memops;

#    ifdef AFS_DAEMON_TRACK_TIME
            /// Whether the tacker has been stopped
            bool stopped;
            /// Name of the function being tracked
            const std::string func_name;
            /// Instant of the construction
            const std::chrono::time_point<std::chrono::system_clock> start;

            /// Start a tracker
            tracker(const std::string &func_name)
                : memops(0),
                  stopped(false),
                  func_name(func_name),
                  start(std::chrono::system_clock::now()) {}

            ~tracker() { stop(); }

            /// Stop the tracker and store the timing
            void stop() {
                if (stopped) return;
                stopped = true;

                // Count elapsed time since the creation of the object
                double elapsed_time =
                    std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();

                // Record the time
                auto &timing = getTimings()[func_name];
                timing.cpu_time += elapsed_time;
                timing.memops += memops;
                timing.calls++;
            }
#    else
            tracker(const std::string &) {}
            void stop() {}
#    endif

            // Forbid copy constructor and assignment operator
            tracker(const tracker &) = delete;
            tracker &operator=(tracker const &) = delete;
        };

        /// Report all tracked timings
        /// \param s: stream to write the report

        template <typename OStream> void reportTimings(OStream &s) {
            // Print the timings alphabetically
            s << "Timing of kernels:" << std::endl;
            s << "------------------" << std::endl;
            const auto &timings = getTimings();
            std::vector<std::string> names;
            for (const auto &it : timings) names.push_back(it.first);
            std::sort(names.begin(), names.end());

            for (const auto &name : names) {
                // Gather the metrics for a given function on all sessions
                auto metric = getTimings().at(name);
                auto time = metric.cpu_time;
                auto memops = metric.memops;
                auto calls = metric.calls;

                double gmemops_per_sec = (time > 0 ? memops / time : 0) / 1024.0 / 1024.0 / 1024.0;
                s << name << " : " << std::fixed << std::setprecision(3) << time << " s ("
                  << "calls: " << std::setprecision(0) << calls //
                  << " bytes: " << memops                       //
                  << " GBYTES/s: " << gmemops_per_sec           //
                  << " )" << std::endl;
            }
            s << std::defaultfloat;
        }

        /// Channel between two threads

        template <typename T> struct Buffer {
        private:
            // Buffer fields
            std::vector<T> buffer;
            std::size_t buffer_size;      ///< active buffer size
            std::size_t first_free_index; ///< first free element
            std::size_t first_element;    ///< first active element in the buffer

            // Fields for concurrency
            std::mutex mtx;
            std::condition_variable not_empty;
            std::condition_variable not_full;

        public:
            Buffer(std::size_t buffer_max_size = 16)
                : buffer(buffer_max_size), buffer_size(0), first_free_index(0), first_element(0) {}

            void push_back(const T &item) {
                // Acquire a unique lock on the mutex
                std::unique_lock<std::mutex> unique_lock(mtx);

                // Wait if the buffer is full
                not_full.wait(unique_lock, [this]() { return buffer_size != buffer.size(); });

                // Add input to buffer
                buffer[first_free_index] = item;

                // Update appropriate fields
                first_free_index = (first_free_index + 1) % buffer.size();
                buffer_size++;

                // Unlock unique lock
                unique_lock.unlock();

                // Notify a single thread that buffer isn't empty
                not_empty.notify_one();
            }

            T pop_front() {
                // Acquire a unique lock on the mutex
                std::unique_lock<std::mutex> unique_lock(mtx);

                // Wait if buffer is empty
                not_empty.wait(unique_lock, [this]() { return buffer_size != 0; });

                // Get value from position to remove in buffer
                const auto result = buffer[first_element];

                // Update appropriate fields
                first_element = (first_element + 1) % buffer.size();
                buffer_size--;

                // Unlock unique lock
                unique_lock.unlock();

                // Notify a single thread that the buffer isn't full
                not_full.notify_one();

                // Return result
                return result;
            }

            std::size_t size() const { return buffer_size; }
        };

        using Func = std::function<void()>;

        inline Buffer<Func> &get_func_buffer() {
            static Buffer<Func> buffer(1024 * 16);
            return buffer;
        }

        enum class Action : int {
            GetFileStatusRequest = 0,
            /// Package description:
            /// - request_num:uint32
            /// - path:null-ending string

            GetFileStatusAnswer = 1,
            /// Package description:
            /// - request_num:uint32
            /// - type:FileType
            /// - file_size:size_t

            GlobalOpenRequest = 2,
            /// Package description:
            /// - request_num:uint32
            /// - file_id:uint32
            /// - path:null-ending string

            GlobalOpenAnswer = 3,
            /// Package description:
            /// - request_num:uint32
            /// - file_size:size_t

            ReadRequest = 4,
            /// Package description:
            /// - request_num:uint32
            /// - file_id:uint32
            /// - offset:size_t
            /// - size:size_t

            ReadAnswer = 5,
            /// Package description:
            /// - request_num:uint32 (part of tag, not the message)
            /// - content:char[*]

            CloseRequest = 6,
            /// Package description:
            /// - request_num:uint32
            /// - file_id:uint32

            GetDirectoryListRequest = 7,
            /// Package description:
            /// - request_num:uint32
            /// - path:null-ending string

            GetDirectoryListAnswer = 8
            /// Package description:
            /// request_num:uint32
            /// [
            ///  - type:FileType
            ///  - path:null-ending string
            /// ]*
        };

        const int MaxAction = 16;

        struct MPI_RequestBuffer {
            MPI_Request request;          ///< MPI request handler
            std::shared_ptr<char> buffer; ///< buffer associated to the request
        };

        /// Return the list of pending requests
        /// NOTE: not thread-safe, accessed only by the MPI loop thread

        inline std::vector<MPI_RequestBuffer> &get_pending_mpi_requests() {
            static std::vector<MPI_RequestBuffer> requests;
            return requests;
        }

        /// Return the number of processes (one process for each filesystem)
        /// Read access by everyone and write access by MPI loop thread

        inline unsigned int &get_num_procs() {
            static unsigned int num_procs;
            return num_procs;
        }

        /// Return this process id (use as id for this filesystem)
        /// Read access by everyone and write access by MPI loop thread

        inline unsigned int &get_proc_id() {
            static unsigned int proc_id;
            return proc_id;
        }

        /// Replace "@NPROC" by the process id in the given string; used for debugging
        /// \param path: given string

        inline std::string replace_hack(const std::string &path) {
            int this_proc = get_proc_id();
            std::string::size_type n = 0;
            std::string re("@NPROC");
            std::string this_proc_s = std::to_string(this_proc);
            std::string path_out = path;
            while ((n = path_out.find(re)) != std::string::npos) {
                path_out.replace(n, re.size(), this_proc_s);
            }
            return path_out;
        }

        /// Type of the request numbers

        using RequestNum = unsigned int;

        /// Read the request number from the string
        /// \param s: pointer to the first element to read

        inline RequestNum get_request_num(const char *s) { return read_from_chars<RequestNum>(s); }

        /// Write the request number into a string
        /// \param req: request number
        /// \param s: pointer to the first element to write into

        inline void set_request_num(RequestNum req, char *s) { write_as_chars(req, s); }

        /// Return a mutex for all the promises

        inline std::mutex &get_talker_mutex() {
            static std::mutex m;
            return m;
        }

        /// Return a condition variable for all the promises

        inline std::condition_variable &get_talker_condition_variable() {
            static std::condition_variable cv;
            return cv;
        }

        /// Simple version of a promise

        template <typename T> struct Promise {
        private:
            T value;
            bool done;

        public:
            /// Promise creation with an initial value
            /// \param value: initial value of the promise

            Promise(const T &value) : value(value), done(false) {}

            /// Promise creation with default initial value

            Promise() : done(false) {}

            /// Set the value of the promise
            /// \param v: given value
            /// NOTE: invoked only by MPI loop thread

            void set(const T &v) {
                value = v;
                done = true;
                get_talker_condition_variable().notify_one();
            }

            /// Wait and return the value given in function `set`
            /// NOTE: invoked only by clients

            const T &get() const {
                std::unique_lock<std::mutex> unique_lock(get_talker_mutex());
                get_talker_condition_variable().wait(unique_lock, [this]() { return done; });
                return value;
            }

            /// Return the current value of the promise; it may be in a inconsistent state

            const T &get_value_unsafe() const { return value; }
        };
    }

    ///
    /// Get information of a file
    ///

    namespace detail {
        namespace detail_get_file_status {

            /// Return the mutex for accessing `get_file_status_pending_transactions`.

            inline std::mutex &get_file_status_pending_transactions_mutex() {
                static std::mutex m;
                return m;
            }

            struct FileTypeAndSize {
                FileType file_type;
                Offset file_size;
            };

            inline std::unordered_map<RequestNum, Promise<FileTypeAndSize>> &
            get_file_status_pending_transactions() {
                static std::unordered_map<RequestNum, Promise<FileTypeAndSize>> pending(16);
                return pending;
            }

            inline void response_file_status_request(int rank, const char *buffer,
                                                     int message_size) {
                RequestNum request_num = get_request_num(buffer);
                std::string path(buffer + sizeof(RequestNum), buffer + message_size);
                std::string path_hack = replace_hack(path);
                log("getting requesting get_file_status from %d: %s\n", rank, path_hack.c_str());

                std::size_t response_buffer_size =
                    sizeof(RequestNum) + sizeof(FileType) + sizeof(Offset);
                auto response_buffer = std::shared_ptr<char>(new char[response_buffer_size],
                                                             std::default_delete<char[]>());
                auto response = response_buffer.get();

                set_request_num(request_num, &response[0]);

                struct stat st;
                memset(&st, 0, sizeof(struct stat));
                int res = stat(path_hack.c_str(), &st);
                FileType file_type = FileType::NotExists;
                Offset file_size = 0;
                if (res != -1) {
                    file_size = st.st_size;
                    switch (st.st_mode & S_IFMT) {
                    case S_IFDIR: file_type = FileType::Directory; break;
                    case S_IFLNK: file_type = FileType::Link; break;
                    case S_IFREG: file_type = FileType::RegularFile; break;
                    default: file_type = FileType::Other; break;
                    }
                }
                write_as_chars(file_type, &response[sizeof(RequestNum)]);
                write_as_chars(file_size, &response[sizeof(RequestNum) + sizeof(FileType)]);

                MPI_Request req;
                check_mpi(MPI_Isend(response, response_buffer_size, MPI_CHAR, rank,
                                    (int)Action::GetFileStatusAnswer, MPI_COMM_WORLD, &req));
                get_pending_mpi_requests().push_back(MPI_RequestBuffer{req, response_buffer});
            }

            inline void response_file_status_answer(int rank, const char *buffer,
                                                    int message_size) {
                assert(message_size == sizeof(RequestNum) + sizeof(FileType) + sizeof(Offset));
                RequestNum request_num = get_request_num(buffer);
                FileType file_type = read_from_chars<FileType>(buffer + sizeof(RequestNum));
                Offset file_size =
                    read_from_chars<Offset>(buffer + sizeof(RequestNum) + sizeof(FileType));
                std::unique_lock<std::mutex> unique_lock(
                    get_file_status_pending_transactions_mutex());
                get_file_status_pending_transactions()
                    .at(request_num)
                    .set(FileTypeAndSize{file_type, file_size});
            }

            inline FileType merge_file_types(FileType a, FileType b) {
                if (a == FileType::NotExists) return b;
                if (b == FileType::NotExists) return a;
                if (a == FileType::Link) return b;
                if (b == FileType::Link) return a;
                if (a != b) return FileType::Other;
                return a;
            }
        }
    }

    /// Get status of a file
    /// \param path: path of the file to inspect
    /// \return: file handle

    inline bool get_status(const char *path, FileType *file_type, Offset *file_size) {
        using namespace detail;
        using namespace detail_get_file_status;

        log("get_status %s\n", path);

        // Mark the requests from this node
        static RequestNum next_req = 0;
        RequestNum first_req = next_req;
        next_req += get_num_procs();

        // Prepare the responses
        {
            std::unique_lock<std::mutex> unique_lock(get_file_status_pending_transactions_mutex());
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank)
                get_file_status_pending_transactions()[first_req + rank] = {};
        }

        // Queue the requests
        std::string msg_pattern = std::string(sizeof(RequestNum), 0) + std::string(path);
        get_func_buffer().push_back([=]() {
            log("send file_status requests %s\n", msg_pattern.c_str() + sizeof(RequestNum));
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                auto this_msg_pattern_buffer = std::shared_ptr<char>(new char[msg_pattern.size()],
                                                                     std::default_delete<char[]>());
                std::copy_n(msg_pattern.begin(), msg_pattern.size(), this_msg_pattern_buffer.get());
                set_request_num(first_req + rank, this_msg_pattern_buffer.get());
                MPI_Request req;
                check_mpi(MPI_Isend(this_msg_pattern_buffer.get(), msg_pattern.size(), MPI_CHAR,
                                    rank, (int)Action::GetFileStatusRequest, MPI_COMM_WORLD, &req));
                get_pending_mpi_requests().push_back(
                    MPI_RequestBuffer{req, this_msg_pattern_buffer});
            }
        });

        // Wait for the responses
        *file_size = 0;
        *file_type = FileType::NotExists;
        auto &pending_transactions = get_file_status_pending_transactions();
        for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
            auto file_type_and_size = pending_transactions.at(first_req + rank).get();
            if (file_type_and_size.file_type == FileType::NotExists) continue;
            *file_type = merge_file_types(*file_type, file_type_and_size.file_type);
            *file_size += file_type_and_size.file_size;
            std::unique_lock<std::mutex> unique_lock(get_file_status_pending_transactions_mutex());
            pending_transactions.erase(first_req + rank);
        }

        // Return whether the file exists
        return *file_type != FileType::NotExists;
    }

    ///
    /// Get directory list
    ///

    namespace detail {
        namespace detail_get_directory_list {
            inline std::mutex &get_directory_list_pending_transactions_mutex() {
                static std::mutex m;
                return m;
            }

            inline std::unordered_map<RequestNum, Promise<std::vector<FilenameType>>> &
            get_directory_list_pending_transactions() {
                static std::unordered_map<RequestNum, Promise<std::vector<FilenameType>>> pending(
                    16);
                return pending;
            }

            inline void response_get_directory_list_request(int rank, const char *buffer,
                                                            int message_size) {
                RequestNum request_num = get_request_num(buffer);
                std::string path(buffer, buffer + message_size);
                log("getting requesting get_directory_list from %d: %s\n", rank, path.c_str());

                std::vector<char> response(sizeof(RequestNum));
                set_request_num(request_num, response.data());

                std::string path_hack = replace_hack(path);
                DIR *dp = opendir(path_hack.c_str());
                if (dp != NULL) {
                    struct dirent *de;
                    while ((de = readdir(dp)) != NULL) {
                        FileType file_type = FileType::Other;
                        switch (de->d_type) {
                        case DT_DIR: // This is a directory.
                            file_type = FileType::Directory;
                            break;
                        case DT_LNK: //  This is a symbolic link.
                            file_type = FileType::Link;
                            break;
                        case DT_REG: // This is a regular file.
                            file_type = FileType::RegularFile;
                            break;
                        }
                        response.push_back((char)file_type);
                        for (int i = 0, name_len = std::strlen(de->d_name); i <= name_len; ++i)
                            response.push_back(de->d_name[i]);
                    }
                    closedir(dp);
                }

                auto response_buffer =
                    std::shared_ptr<char>(new char[response.size()], std::default_delete<char[]>());
                std::copy(response.begin(), response.end(), response_buffer.get());
                MPI_Request req;
                check_mpi(MPI_Isend(response_buffer.get(), response.size(), MPI_CHAR, rank,
                                    (int)Action::GetDirectoryListAnswer, MPI_COMM_WORLD, &req));
                get_pending_mpi_requests().push_back(MPI_RequestBuffer{req, response_buffer});
            }

            inline void response_get_directory_list_answer(int rank, const char *buffer,
                                                           int message_size) {
                RequestNum request_num = get_request_num(buffer);
                const char *msg_it = buffer + sizeof(RequestNum);
                std::vector<FilenameType> response;
                for (const char *msg_end = buffer + message_size; msg_it != msg_end;) {
                    // Get type
                    FileType type = (FileType)*msg_it++;
                    // Get name
                    const char *filename = msg_it;
                    unsigned int filename_len = std::strlen(filename);
                    response.push_back(
                        FilenameType{std::string(filename, filename + filename_len), type});
                    msg_it += filename_len + 1;
                    log("getting answer get_directory_list from %d: %s\n", rank, filename);
                }
                std::unique_lock<std::mutex> unique_lock(
                    get_directory_list_pending_transactions_mutex());
                get_directory_list_pending_transactions().at(request_num).set(response);
            }
        }
    }

    /// Return the list of files and directories under a given path among all filesystems
    /// \param path: pathname relative to the roots of the filesystems
    /// \return: list of filenames and types

    inline std::vector<FilenameType> get_directory_list(const char *path) {
        using namespace detail;
        using namespace detail_get_directory_list;

        log("get_directory_list %s\n", path);

        // Mark the requests from this node
        static RequestNum next_req = 0;
        RequestNum first_req = next_req;
        next_req += get_num_procs();

        // Prepare the responses
        {
            std::unique_lock<std::mutex> unique_lock(
                get_directory_list_pending_transactions_mutex());
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank)
                get_directory_list_pending_transactions()[first_req + rank] = {};
        }

        // Queue the requests
        std::string msg_pattern = std::string(sizeof(RequestNum), 0) + std::string(path);
        get_func_buffer().push_back([=]() {
            log("send get_directory_list requests %s\n", msg_pattern.c_str() + sizeof(RequestNum));
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                auto this_msg_pattern_buffer = std::shared_ptr<char>(new char[msg_pattern.size()],
                                                                     std::default_delete<char[]>());
                std::copy_n(msg_pattern.begin(), msg_pattern.size(), this_msg_pattern_buffer.get());
                set_request_num(first_req + rank, this_msg_pattern_buffer.get());
                MPI_Request req;
                check_mpi(MPI_Isend(this_msg_pattern_buffer.get(), msg_pattern.size(), MPI_CHAR,
                                    rank, (int)Action::GetDirectoryListRequest, MPI_COMM_WORLD,
                                    &req));
                get_pending_mpi_requests().push_back(
                    MPI_RequestBuffer{req, this_msg_pattern_buffer});
            }
        });

        // Wait for the responses
        std::unordered_map<std::string, FileType> response(16);
        auto &pending_transactions = get_directory_list_pending_transactions();
        for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
            auto response_rank = pending_transactions.at(first_req + rank).get();
            for (const auto &it : response_rank) {
                if (response.count(it.filename) == 0)
                    response[it.filename] = it.type;
                else if (response.at(it.filename) != it.type)
                    response[it.filename] = FileType::Other;
            }
            std::unique_lock<std::mutex> unique_lock(
                get_directory_list_pending_transactions_mutex());
            pending_transactions.erase(first_req + rank);
        }

        std::vector<FilenameType> r;
        r.reserve(response.size());
        for (const auto &it : response) r.push_back(FilenameType{it.first, it.second});
        return r;
    }

    ///
    /// Global open file
    ///

    namespace detail {

#    ifdef AFS_DAEMON_USE_MPIIO
        using FileHandle = MPI_File;

        inline bool file_open(const char *filename, MPI_File &f) {
            return MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f) ==
                   MPI_SUCCESS;
        }

        inline Offset get_file_size(const MPI_File &f) {
            MPI_Offset offset;
            check_mpi(MPI_File_get_size(f, &offset));
            return offset;
        }

        inline MPI_Request file_read(const MPI_File &f, std::size_t offset, char *v,
                                     std::size_t n) {
            MPI_Request req;
            check_mpi(MPI_File_seek(f, offset, MPI_SEEK_SET));
            check_mpi(MPI_File_iread(f, v, n, MPI_CHAR, &req));
            return req;
        }

        inline void file_close(MPI_File &f) { check_mpi(MPI_File_close(&f)); }
#    else
        using FileHandle = std::FILE *;

        inline bool file_open(const char *filename, std::FILE *&f) {
            f = std::fopen(filename, "r");
            return f != NULL;
        }

        inline Offset get_file_size(std::FILE *f) {
            // Get the current size of the file
            off_t end_of_file;
            if (std::fseek(f, -1, SEEK_END) != 0)
                throw std::runtime_error("Error setting file position");

            if ((end_of_file = std::ftell(f) + 1) == 0)
                throw std::runtime_error("Error getting file position");

            Offset curr_size = end_of_file;
            return curr_size;
        }

        inline void file_read(std::FILE *f, std::size_t offset, char *v, std::size_t n) {
            if (std::fseek(f, offset, SEEK_SET) == 0) {
                std::size_t count = std::fread(v, sizeof(char), n, f);
                if (count == n) return;
            }
            throw std::runtime_error("file_read: error while reading");
        }

        inline void file_close(std::FILE *f) { std::fclose(f); }
#    endif

        using FileInfo = std::vector<Offset>; ///< First offset on each node

        using OpenFilesCache = std::unordered_map<FileId, FileInfo>;

        inline OpenFilesCache &get_open_files_cache() {
            static OpenFilesCache open_files_cache(16);
            return open_files_cache;
        }

        struct FromAndFileId {
            int from;
            FileId file_id;

            bool operator==(const FromAndFileId &f) const {
                return from == f.from && file_id == f.file_id;
            }
        };

        // Hash function for FromAndFileId
        struct HashForFromAndFileId {
            std::size_t operator()(const FromAndFileId &s) const noexcept {
                std::size_t h1 = std::hash<int>{}(s.from);
                std::size_t h2 = std::hash<FileId>{}(s.file_id);
                return h1 ^ (h2 << 1);
            }
        };

        struct LocalOpenedFiles {
            /// From path to file handler
            /// From path to counts
            struct HandlerAndCount {
                FileHandle f;
                unsigned int count;
            };
            std::unordered_map<std::string, HandlerAndCount> from_path_to_handler_and_count;

            /// From file id to path
            /// From file id to file handler
            struct PathAndHandler {
                std::string path;
                FileHandle f;
            };
            std::unordered_map<FromAndFileId, PathAndHandler, HashForFromAndFileId>
                from_file_id_to_path_and_handler;

            LocalOpenedFiles()
                : from_path_to_handler_and_count(16), from_file_id_to_path_and_handler(16) {}

            bool open(const char *path, const FromAndFileId &file_id, FileHandle &f) {
                std::string path_s(path);
                if (from_path_to_handler_and_count.count(path) == 0) {
                    if (!file_open(path, f)) return false;
                    from_path_to_handler_and_count[path_s] = {f, 1};
                } else {
                    auto &handler_and_count = from_path_to_handler_and_count[path_s];
                    f = handler_and_count.f;
                    handler_and_count.count++;
                }
                from_file_id_to_path_and_handler[file_id] = {path_s, f};
                return true;
            }

            bool get_file_handler(const FromAndFileId &file_id, FileHandle &f) {
                if (from_file_id_to_path_and_handler.count(file_id) == 0) return false;
                f = from_file_id_to_path_and_handler.at(file_id).f;
                return true;
            }

            bool close(const FromAndFileId &file_id) {
                if (from_file_id_to_path_and_handler.count(file_id) == 0) return false;
                auto path_and_handler = from_file_id_to_path_and_handler.at(file_id);
                auto handler_and_count = from_path_to_handler_and_count.at(path_and_handler.path);
                if (handler_and_count.count == 1) {
                    file_close(handler_and_count.f);
                    from_path_to_handler_and_count.erase(path_and_handler.path);
                } else {
                    from_path_to_handler_and_count.at(path_and_handler.path).count--;
                }
                from_file_id_to_path_and_handler.erase(file_id);
                return true;
            }
        };

        /// Execute a callback only after a number of calls

        enum class QueueCallback { DoQueueCallback, DontQueueCallback };

        template <QueueCallback queue_callback> struct TickingCallback {
            /// Actual callback to execute
            std::shared_ptr<std::function<void()>> callback;

            TickingCallback() {}

            TickingCallback(const std::function<void()> &callback)
                : callback(std::make_shared<std::function<void()>>(callback)) {}

            /// Attempt to execute the callback
            void call() {
                // If there is only one reference, queue the callback
                if (callback.use_count() == 1) {
                    if (queue_callback == QueueCallback::DoQueueCallback) {
                        std::function<void()> f = *callback;
                        get_func_buffer().push_back([=]() { f(); });
                    } else {
                        (*callback)();
                    }
                }
                callback.reset();
            }
        };

        inline LocalOpenedFiles &get_local_opened_files() {
            static LocalOpenedFiles local_opened_files{};
            return local_opened_files;
        }

        namespace detail_open_file {
            struct SizeAndCallback {
                std::size_t size;
                TickingCallback<QueueCallback::DoQueueCallback> callback;
            };

            /// Get open file pending transactions
            /// NOTE: Access only by MPI loop thread

            inline std::unordered_map<RequestNum, SizeAndCallback> &
            get_open_file_pending_transactions() {
                static std::unordered_map<RequestNum, SizeAndCallback> pending(16);
                return pending;
            }

            inline void response_global_open_file_request(int rank, const char *buffer,
                                                          int message_size) {
                tracker t_("open file processing requests");

                RequestNum request_num = get_request_num(buffer);
                FileId file_id = read_from_chars<FileId>(buffer + sizeof(RequestNum));
                std::string path(buffer + sizeof(RequestNum) + sizeof(FileId),
                                 buffer + message_size);
                std::string path_hack = replace_hack(path);
                log("mpi open id: %d process response from: %d: file: %s\n", (int)file_id, rank,
                    path_hack.c_str());

                std::size_t response_buffer_size = sizeof(RequestNum) + sizeof(Offset);
                auto response_buffer = std::shared_ptr<char>(new char[response_buffer_size],
                                                             std::default_delete<char[]>());
                auto response = response_buffer.get();

                set_request_num(request_num, &response[0]);

                FileHandle f;
                bool success = get_local_opened_files().open(path_hack.c_str(),
                                                             FromAndFileId{rank, file_id}, f);
                Offset file_size_plus_one = 0;
                if (success) file_size_plus_one = get_file_size(f) + 1;
                write_as_chars(file_size_plus_one, &response[sizeof(RequestNum)]);

                MPI_Request req;
                check_mpi(MPI_Isend(response, response_buffer_size, MPI_CHAR, rank,
                                    (int)Action::GlobalOpenAnswer, MPI_COMM_WORLD, &req));
                get_pending_mpi_requests().push_back(MPI_RequestBuffer{req, response_buffer});
            }

            inline void response_global_open_file_answer(int rank, const char *buffer,
                                                         int message_size) {
                tracker t_("open file processing answers");

                RequestNum request_num = get_request_num(buffer);
                const char *msg_it = buffer + sizeof(RequestNum);
                Offset file_size_plus_one = read_from_chars<Offset>(msg_it);
                auto &size_and_callback = get_open_file_pending_transactions().at(request_num);
                size_and_callback.size = file_size_plus_one;
                size_and_callback.callback.call();
            }
        }
    }

    /// Open a file
    /// \param path: path of the file to open
    /// \param: callback with the file handle

    inline void get_open_file(const char *path, std::function<void(FileId)> response_callback) {
        using namespace detail;
        using namespace detail_open_file;

        detail::tracker t_("open file api");

        static RequestNum next_req = 0;
        RequestNum first_req = next_req;
        next_req += get_num_procs();

        // Create an entry
        static FileId next_file_id = 1;
        if (next_file_id == no_file_id) next_file_id++;
        FileId file_id = next_file_id++;

        log("open id: %d (starting) file: %s\n", (int)file_id, path);

        // Mark the requests from this node
        // Queue the requests
        std::string msg_pattern =
            std::string(sizeof(RequestNum) + sizeof(FileId), 0) + std::string(path);
        write_as_chars(file_id, &msg_pattern[sizeof(RequestNum)]);
        const auto sender = [=](std::function<void()> process) {
            detail::tracker t_("open file sending requests");

            // Prepare the responses
            TickingCallback<QueueCallback::DoQueueCallback> callback(process);
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank)
                get_open_file_pending_transactions().emplace(
                    std::make_pair(first_req + rank, SizeAndCallback{0, callback}));

            // Send the requests
            //log("send get_open_file requests %d\n", file_id);
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                auto this_msg_pattern_buffer = std::shared_ptr<char>(new char[msg_pattern.size()],
                                                                     std::default_delete<char[]>());
                std::copy_n(msg_pattern.begin(), msg_pattern.size(), this_msg_pattern_buffer.get());
                set_request_num(first_req + rank, this_msg_pattern_buffer.get());
                MPI_Request req;
                check_mpi(MPI_Isend(this_msg_pattern_buffer.get(), msg_pattern.size(), MPI_CHAR,
                                    rank, (int)Action::GlobalOpenRequest, MPI_COMM_WORLD, &req));
                get_pending_mpi_requests().push_back(
                    MPI_RequestBuffer{req, this_msg_pattern_buffer});
            }
        };

        // Process responses
        const auto process = [=]() {
            log("mpi open id: %d (process)\n", (int)file_id);

            std::vector<Offset> file_sizes(get_num_procs());
            bool file_exists = false;
            auto &pending_transactions = get_open_file_pending_transactions();
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                Offset file_size_plus_one = pending_transactions.at(first_req + rank).size;
                if (file_size_plus_one > 0) file_exists = true;
                file_sizes[rank] = file_size_plus_one == 0 ? 0 : file_size_plus_one - 1;
            }

            // Erase transactions
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank)
                pending_transactions.erase(first_req + rank);

            // Return special code if the file does not exists
            if (!file_exists) {
                log("open request for file_id %d does not exist\n", (int)file_id);
                response_callback(no_file_id);
                return;
            }

            // Get the offsets
            std::vector<Offset> offsets(get_num_procs() + 1);
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank)
                offsets[rank + 1] = offsets[rank] + file_sizes[rank];
            get_open_files_cache()[file_id] = offsets;

            response_callback(file_id);
        };

        // Execute
        get_func_buffer().push_back([=]() { sender(process); });
    }

    /// Open a file
    /// \param path: path of the file to open
    /// \return: file handle

    inline FileId get_open_file(const char *path) {
        struct Void {};
        detail::Promise<Void> promise{};
        FileId file_id;
        get_open_file(path, [&](FileId f) {
            file_id = f;
            promise.set({});
        });
        promise.get();
        return file_id;
    }

    ///
    /// Read from opened file
    ///

    namespace detail {
        struct MPI_RequestCallback {
            MPI_Request request; ///< MPI request handler
            TickingCallback<QueueCallback::DontQueueCallback>
                callback; ///< callback associated to the request
        };

        /// Return the list of pending requests
        /// NOTE: not thread-safe, accessed only by the MPI loop thread

        inline std::vector<MPI_RequestCallback> &get_pending_mpi_request_callbacks() {
            static std::vector<MPI_RequestCallback> requests;
            return requests;
        }

        namespace detail_read {
            inline void response_read_request(int rank, const char *buffer, int message_size) {
                tracker t_("read file processing requests");

                RequestNum request_num = get_request_num(buffer);
                FileId file_id = read_from_chars<FileId>(buffer + sizeof(RequestNum));
                Offset local_offset =
                    read_from_chars<Offset>(buffer + sizeof(RequestNum) + sizeof(FileId));
                Offset local_size = read_from_chars<Offset>(buffer + sizeof(RequestNum) +
                                                            sizeof(FileId) + sizeof(Offset));
                log("mpi read request: %d id: %d from: %d size: %d\n", (int)request_num,
                    (int)file_id, (int)local_offset, (int)local_size);

                auto response_buffer =
                    std::shared_ptr<char>(new char[local_size], std::default_delete<char[]>());

                tracker t0_("read file processing requests (file_read)");
                FileHandle f;
                if (!get_local_opened_files().get_file_handler(FromAndFileId{rank, file_id}, f))
                    throw std::runtime_error("response_read_request: file_id is not a valid");
#    ifdef AFS_DAEMON_USE_MPIIO
                MPI_Request req = file_read(f, local_offset, response_buffer.get(), local_size);
                get_pending_mpi_request_callbacks().push_back(MPI_RequestCallback{
                    req, TickingCallback<QueueCallback::DontQueueCallback>([=]() {
                        tracker t_("read file processing requests (MPI_Isend)");
                        MPI_Request req;
                        check_mpi(MPI_Isend(response_buffer.get(), local_size, MPI_CHAR, rank,
                                            (int)Action::ReadAnswer + (int)request_num * MaxAction,
                                            MPI_COMM_WORLD, &req));
                        get_pending_mpi_requests().push_back(
                            MPI_RequestBuffer{req, response_buffer});
                    })});

#    else
                file_read(f, local_offset, response_buffer.get(), local_size);
                t0_.stop();

                tracker t1_("read file processing requests (MPI_Isend)");
                MPI_Request req;
                check_mpi(MPI_Isend(response_buffer.get(), local_size, MPI_CHAR, rank,
                                    (int)Action::ReadAnswer + (int)request_num * MaxAction,
                                    MPI_COMM_WORLD, &req));
                get_pending_mpi_requests().push_back(MPI_RequestBuffer{req, response_buffer});
#    endif
            }
        }
    }

    /// Read from an open file
    /// \param file_id: file handler to read from
    /// \param offset: first character to read
    /// \param count: number of characters to read
    /// \param buffer: memory pointer where to write the content
    /// \param: callback with the file handle

    inline void read(FileId file_id, std::size_t offset, std::size_t count, char *buffer,
                     std::function<void(std::int64_t)> response_callback) {
        using namespace detail;
        using namespace detail_read;

        detail::tracker t_("read file api");

        // Quick answer if the count is zero
        if (count == 0) {
            response_callback(0);
            return;
        }

        // Mark the requests from this node
        static RequestNum next_req = 0;
        RequestNum first_req = next_req;
        next_req += get_num_procs();

        log("mpi read (starting) request: %d id: %d from: %d size: %d\n", (int)first_req,
            (int)file_id, (int)offset, (int)count);

        // Queue the requests
        auto send = [=](const std::function<void()> &process) {
            detail::tracker t_("read file sending requests");

            // Quick answer if the file_id does not exists
            if (get_open_files_cache().count(file_id) == 0) {
                response_callback(-1);
                return;
            }

            // Prepare the responses
            static std::vector<int> next_tag_request_number(get_num_procs());
            std::vector<Offset> local_offsets(get_num_procs());
            std::vector<Offset> local_counts(get_num_procs());
            std::vector<Offset> str_offsets(get_num_procs());
            const auto &offsets = get_open_files_cache().at(file_id);
            TickingCallback<QueueCallback::DontQueueCallback> callback(process);
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                Offset first_element = std::max(offsets[rank], std::min(offsets[rank + 1], offset));
                Offset last_element =
                    std::max(offsets[rank], std::min(offsets[rank + 1], offset + count));
                if (first_element < last_element) {
                    str_offsets[rank] = first_element - offset;
                    local_offsets[rank] = first_element - offsets[rank];
                    local_counts[rank] = last_element - first_element;
                    MPI_Request req;
                    check_mpi(MPI_Irecv(
                        buffer + str_offsets[rank], local_counts[rank], MPI_CHAR, rank,
                        (int)Action::ReadAnswer + MaxAction * next_tag_request_number[rank],
                        MPI_COMM_WORLD, &req));
                    get_pending_mpi_request_callbacks().push_back(
                        MPI_RequestCallback{req, callback});
                }
            }

            //log("mpi read (send) request: %d\n", (int)first_req);
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                if (local_counts[rank] == 0) continue;
                std::size_t this_msg_pattern_size =
                    sizeof(RequestNum) + sizeof(FileId) + sizeof(Offset) * 2;
                auto this_msg_pattern_buffer = std::shared_ptr<char>(
                    new char[this_msg_pattern_size], std::default_delete<char[]>());
                auto this_msg_pattern = this_msg_pattern_buffer.get();
                set_request_num(RequestNum(next_tag_request_number[rank]), &this_msg_pattern[0]);
                write_as_chars(file_id, &this_msg_pattern[sizeof(RequestNum)]);
                write_as_chars(local_offsets[rank],
                               &this_msg_pattern[sizeof(RequestNum) + sizeof(FileId)]);
                write_as_chars(
                    local_counts[rank],
                    &this_msg_pattern[sizeof(RequestNum) + sizeof(FileId) + sizeof(Offset)]);
                MPI_Request req;
                check_mpi(MPI_Isend(this_msg_pattern, this_msg_pattern_size, MPI_CHAR, rank,
                                    (int)Action::ReadRequest, MPI_COMM_WORLD, &req));
                get_pending_mpi_requests().push_back(
                    MPI_RequestBuffer{req, this_msg_pattern_buffer});
                next_tag_request_number[rank] = (next_tag_request_number[rank] + 1) %
                                                (std::numeric_limits<int>::max() / MaxAction);
            }
        };

        // Process responses
        const auto process = [=]() {
            log("mpi read (process) request: %d\n", (int)first_req);

            detail::tracker t_("read file finalizing requests");
            response_callback(count);
        };

        // Execute
        get_func_buffer().push_back([=]() { send(process); });
    }

    /// Read from an open file
    /// \param file_id: file handler to read from
    /// \param offset: first character to read
    /// \param count: number of characters to read
    /// \param buffer: memory pointer where to write the content
    /// \return: if positive, the number of read characters; otherwise, error code

    inline std::int64_t read(FileId file_id, std::size_t offset, std::size_t count, char *buffer) {
        struct Void {};
        detail::Promise<Void> promise{};
        std::int64_t read_chars_or_error = 0;
        read(file_id, offset, count, buffer, [&](std::int64_t r) {
            read_chars_or_error = r;
            promise.set({});
        });
        promise.get();
        return read_chars_or_error;
    }

    ///
    /// Closed opened file
    ///

    namespace detail {
        namespace detail_close {
            inline void response_close_request(int rank, const char *buffer, int message_size) {
                assert(message_size == sizeof(RequestNum) + sizeof(FileId));
                RequestNum request_num = get_request_num(buffer);
                (void)request_num;
                FileId file_id = read_from_chars<FileId>(buffer + sizeof(RequestNum));
                log("mpi close (request) id: %d\n", (int)file_id);

                if (!get_local_opened_files().close(FromAndFileId{rank, file_id}))
                    throw std::runtime_error("response_close_request: invalid file_id");
            }
        }
    }

    /// Close an open file
    /// \param file_id: file handler to close
    /// \param: callback with the file handle

    inline void close(FileId file_id, std::function<void(bool)> response_callback) {
        using namespace detail;
        using namespace detail_close;

        log("mpi close id: %d\n", file_id);

        // Mark the requests from this node
        static RequestNum next_req = 0;
        RequestNum first_req = next_req;
        next_req++;

        // Queue the requests
        get_func_buffer().push_back([=]() {
            //log("send close requests for file id %d\n", file_id);

            // Quick answer if the file_id does not exists
            if (get_open_files_cache().count(file_id) == 0) {
                response_callback(false);
                return;
            }

            const auto &offsets = get_open_files_cache().at(file_id);
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                if (offsets[rank] == offsets[rank + 1]) continue;
                std::size_t this_msg_pattern_size = sizeof(RequestNum) + sizeof(FileId);
                auto this_msg_pattern_buffer = std::shared_ptr<char>(
                    new char[this_msg_pattern_size], std::default_delete<char[]>());
                auto this_msg_pattern = this_msg_pattern_buffer.get();
                set_request_num(first_req + rank, &this_msg_pattern[0]);
                write_as_chars(file_id, &this_msg_pattern[sizeof(RequestNum)]);
                MPI_Request req;
                check_mpi(MPI_Isend(this_msg_pattern, this_msg_pattern_size, MPI_CHAR, rank,
                                    (int)Action::CloseRequest, MPI_COMM_WORLD, &req));
                get_pending_mpi_requests().push_back(
                    MPI_RequestBuffer{req, this_msg_pattern_buffer});
            }

            get_open_files_cache().erase(file_id);
            response_callback(true);
        });
    }

    /// Close an open file
    /// \param file_id: file handler to close
    /// \return: true if success

    inline bool close(FileId file_id) {
        struct Void {};
        detail::Promise<Void> promise{};
        std::int64_t success = false;
        close(file_id, [&](bool r) {
            success = r;
            promise.set({});
        });
        promise.get();
        return success;
    }

    ///
    /// Internal messages process
    ///

    namespace detail {
        inline bool process_internal_messages() {
            tracker t_("processing internal MPI messages");
            bool something_done = false;
            while (true) {
                MPI_Message msg;
                MPI_Status status;
                int flag;
                {
                    tracker t_("processing internal MPI messages (MPI_Iprobe)");
                    check_mpi(MPI_Improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &msg,
                                          &status));
                }
                if (flag == 0) break;

                int origin = status.MPI_SOURCE;
                Action action = (Action)(status.MPI_TAG % MaxAction);
                int message_size;
                check_mpi(MPI_Get_count(&status, MPI_CHAR, &message_size));
                static std::vector<char> msg_buffer;
                if (msg_buffer.size() < (std::size_t)message_size) msg_buffer.resize(message_size);
                {
                    tracker t_("processing MPI messages (MPI_Mrecv)");
                    check_mpi(MPI_Mrecv(msg_buffer.data(), message_size, MPI_CHAR, &msg,
                                        MPI_STATUS_IGNORE));
                }
                log("got a message from %d\n", origin);
                switch (action) {
                case Action::GetFileStatusRequest:
                    detail_get_file_status::response_file_status_request(origin, msg_buffer.data(),
                                                                         message_size);
                    break;

                case Action::GetFileStatusAnswer:
                    detail_get_file_status::response_file_status_answer(origin, msg_buffer.data(),
                                                                        message_size);
                    break;

                case Action::GlobalOpenRequest:
                    detail_open_file::response_global_open_file_request(origin, msg_buffer.data(),
                                                                        message_size);
                    break;

                case Action::GlobalOpenAnswer:
                    detail_open_file::response_global_open_file_answer(origin, msg_buffer.data(),
                                                                       message_size);
                    break;

                case Action::GetDirectoryListRequest:
                    detail_get_directory_list::response_get_directory_list_request(
                        origin, msg_buffer.data(), message_size);
                    break;

                case Action::GetDirectoryListAnswer:
                    detail_get_directory_list::response_get_directory_list_answer(
                        origin, msg_buffer.data(), message_size);
                    break;

                case Action::ReadRequest:
                    detail_read::response_read_request(origin, msg_buffer.data(), message_size);
                    break;

                case Action::CloseRequest:
                    detail_close::response_close_request(origin, msg_buffer.data(), message_size);
                    break;

                default: throw std::runtime_error("unexpected action code"); break;
                }
                something_done = true;
            }

            return something_done;
        }
    }

    ///
    /// Client process
    ///

    namespace detail {
        /// Return the list of pending requests
        /// NOTE: not thread-safe, accessed only by the MPI loop thread

        inline std::vector<MPI_Comm> &get_clients() {
            static std::vector<MPI_Comm> clients;
            return clients;
        }

        inline void process_client_action(ClientAction action, const MPI_Comm &comm, int rank,
                                          const char *buffer, int message_size) {
            // Process action
            switch (action) {
            case ClientAction::GlobalOpenRequest: {
                // Get path
                std::string path(buffer, buffer + message_size);

                // Open file
                get_open_file(path.c_str(), [=](FileId file_id) {
                    // Return back the file id
                    auto response_buffer_size = sizeof(FileId);
                    auto response_buffer = std::shared_ptr<char>(new char[response_buffer_size],
                                                                 std::default_delete<char[]>());
                    write_as_chars(file_id, response_buffer.get());
                    MPI_Request req;
                    check_mpi(MPI_Isend(response_buffer.get(), response_buffer_size, MPI_CHAR, rank,
                                        (int)ClientAction::GlobalOpenRequest, comm, &req));
                    get_pending_mpi_requests().push_back(MPI_RequestBuffer{req, response_buffer});
                });
                break;
            }

            case ClientAction::ReadRequest: {
                // Read file id
                FileId file_id = read_from_chars<FileId>(buffer);
                // Read Offset
                Offset offset = read_from_chars<Offset>(buffer + sizeof(FileId));
                // Read size
                Offset size = read_from_chars<Offset>(buffer + sizeof(FileId) + sizeof(Offset));
                // Make the read
                auto read_buffer =
                    std::shared_ptr<char>(new char[size], std::default_delete<char[]>());
                read(file_id, offset, size, read_buffer.get(), [=](std::int64_t size_or_error) {
                    std::size_t response_buffer_size = size;
                    auto response_buffer = read_buffer;
                    int tag = (int)ClientAction::ReadRequest;
                    if (size_or_error < 0) {
                        // Return the error code only in case of error
                        response_buffer_size = sizeof(std::int64_t);
                        response_buffer = std::shared_ptr<char>(new char[response_buffer_size],
                                                                std::default_delete<char[]>());
                        write_as_chars(size_or_error, response_buffer.get());
                        tag = -1;
                    }
                    MPI_Request req;
                    check_mpi(MPI_Isend(response_buffer.get(), response_buffer_size, MPI_CHAR, rank,
                                        tag, comm, &req));
                    get_pending_mpi_requests().push_back(MPI_RequestBuffer{req, response_buffer});
                });
                break;
            }

            case ClientAction::CloseRequest: {
                // Read file id
                FileId file_id = read_from_chars<FileId>(buffer);

                // Close the file
                close(file_id, [=](bool success) {
                    // Return back whether the action was successful
                    auto response_buffer_size = sizeof(std::uint32_t);
                    auto response_buffer = std::shared_ptr<char>(new char[response_buffer_size],
                                                                 std::default_delete<char[]>());
                    write_as_chars(std::uint32_t(success ? 0 : 1), response_buffer.get());
                    MPI_Request req;
                    check_mpi(MPI_Isend(response_buffer.get(), response_buffer_size, MPI_CHAR, rank,
                                        (int)ClientAction::CloseRequest, comm, &req));
                    get_pending_mpi_requests().push_back(MPI_RequestBuffer{req, response_buffer});
                });
                break;
            }

            default: throw std::runtime_error("process_client_action: invalid client action");
            }
        }

        inline bool process_client_messages() {
            tracker t_("processing client MPI messages");
            bool something_done = false;
            for (const auto comm : get_clients()) {
                MPI_Message msg;
                MPI_Status status;
                int flag;
                {
                    tracker t_("processing internal MPI messages (MPI_Iprobe)");
                    check_mpi(MPI_Improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &msg, &status));
                }
                if (flag == 0) continue;

                int origin = status.MPI_SOURCE;
                int message_size;
                ClientAction action = (ClientAction)status.MPI_TAG;
                check_mpi(MPI_Get_count(&status, MPI_CHAR, &message_size));
                static std::vector<char> msg_buffer;
                if (msg_buffer.size() < (std::size_t)message_size) msg_buffer.resize(message_size);
                {
                    tracker t_("processing MPI messages (MPI_Mrecv)");
                    check_mpi(MPI_Mrecv(msg_buffer.data(), message_size, MPI_CHAR, &msg,
                                        MPI_STATUS_IGNORE));
                }
                process_client_action(action, comm, origin, msg_buffer.data(), message_size);
                something_done = true;
            }
            return something_done;
        }
    }

    ///
    /// Main MPI loop
    ///

    namespace detail {
        inline std::thread &get_mpi_thread() {
            static std::thread th;
            return th;
        }

        inline bool &get_finalize_mpi_thread() {
            static bool b = false;
            return b;
        }

        inline bool &is_mpi_initialized() {
            static bool b = false;
            return b;
        }

        inline std::string &get_port_name() {
            static std::string port_name;
            return port_name;
        }

        inline void mpi_loop(int *argc, char **argv[]) {
            bool mpi_is_active = false; // whether mpi is initialized
            try {
                log("mpi thread is active, baby!\n");
                // Initialize MPI
                //int provided = 0;
                //check_mpi(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided));
                //if (provided < MPI_THREAD_FUNNELED)
                //    throw std::runtime_error("MPI does not support the required thread level");
                check_mpi(MPI_Init(argc, argv));
                is_mpi_initialized() = true;
                mpi_is_active = true;

                int nprocs, this_proc;
                check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &this_proc));
                check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
                log("nprocs= %d\n", nprocs);
                get_num_procs() = nprocs;
                get_proc_id() = this_proc;

                // Open a port
                char port_name[MPI_MAX_PORT_NAME];
                check_mpi(MPI_Open_port(MPI_INFO_NULL, port_name));
                get_port_name() = std::string(port_name);

#    ifdef AFS_DAEMON_TRACK_TIME
                std::chrono::time_point<std::chrono::system_clock> last_report =
                    std::chrono::system_clock::now();
#    endif

                auto &buffer = get_func_buffer();
                while (!get_finalize_mpi_thread()) {
                    bool something_done = false;

                    // Check pending MPI requests
                    auto &pending_mpi_requests = get_pending_mpi_requests();
                    if (pending_mpi_requests.size() > 0) {
                        log("checking pending MPI requests\n");
                        tracker t_("checking pending MPI requests");
                        pending_mpi_requests.erase(
                            std::remove_if(pending_mpi_requests.begin(), pending_mpi_requests.end(),
                                           [](MPI_RequestBuffer &req_buffer) {
                                               int flag;
                                               check_mpi(MPI_Test(&req_buffer.request, &flag,
                                                                  MPI_STATUS_IGNORE));
                                               return flag != 0;
                                           }),
                            pending_mpi_requests.end());
                        something_done = true;
                    }

                    // Check pending MPI requests with callback
                    auto &pending_mpi_request_callbacks = get_pending_mpi_request_callbacks();
                    if (pending_mpi_request_callbacks.size() > 0) {
                        log("checking pending MPI requests with callbacks\n");
                        tracker t_("checking pending MPI requests with callbacks");
                        pending_mpi_request_callbacks.erase(
                            std::remove_if(pending_mpi_request_callbacks.begin(),
                                           pending_mpi_request_callbacks.end(),
                                           [](MPI_RequestCallback &req_callback) {
                                               int flag;
                                               check_mpi(MPI_Test(&req_callback.request, &flag,
                                                                  MPI_STATUS_IGNORE));
                                               if (flag != 0) {
                                                   req_callback.callback.call();
                                                   return true;
                                               }
                                               return false;
                                           }),
                            pending_mpi_request_callbacks.end());
                        something_done = true;
                    }

                    // Process client messages
                    if (process_client_messages()) something_done = true;

                    // Process internal messages
                    if (process_internal_messages()) something_done = true;

                    // Do pending tasks
                    {
                        tracker t_("do pending tasks");
                        while (buffer.size() > 0) {
                            log("pending function to execute\n");
                            buffer.pop_front()();
                            something_done = true;
                        }
                    }

#    ifdef AFS_DAEMON_TRACK_TIME
                    const auto now = std::chrono::system_clock::now();
                    if (std::chrono::duration<double>(now - last_report).count() >
                        10 /* 5 mins */) {
                        detail::reportTimings(std::cout);
                        last_report = now;
                    }
#    endif

                    if (!something_done) std::this_thread::yield();
                }
            } catch (const std::exception &e) {
                show_error("Error happened in `anarchofs::mpi_loop`: %s\n", e.what());
            }

            // Finalize MPI
            log("Finalizing MPI\n");
            if (mpi_is_active) {
                check_mpi(MPI_Close_port(get_port_name().c_str()));
                MPI_Finalize();
                is_mpi_initialized() = false;
            }
        }
    }

    /// Start the processing messages loop
    /// \param argc: number of arguments (required by MPI_Init)
    /// \param argv: list of commandline arguments (required by MPI_Init)

    inline bool start_mpi_loop(int *argc, char **argv[]) {
        using namespace detail;
        log("requesting starting MPI loop\n");
        try {
            is_mpi_initialized() = false;
            get_finalize_mpi_thread() = false;
            get_mpi_thread() = std::thread([=]() { mpi_loop(argc, argv); });
            while (!is_mpi_initialized()) std::this_thread::yield();
            return true;
        } catch (const std::exception &e) {
            show_error("Error happened in `anarchofs::start_mpi_loop`: %s\n", e.what());
            return false;
        }
    }

    /// Stop the processing messages loop initiated by `start_mpi_loop`

    inline bool stop_mpi_loop() {
        using namespace detail;
        log("requesting stopping MPI loop\n");
        try {
            get_finalize_mpi_thread() = true;
            get_mpi_thread().join();
            log("MPI loop stopped\n");
            return true;
        } catch (const std::exception &e) {
            show_error("Error happened in `anarchofs::stop_mpi_loop`: %s\n", e.what());
            return false;
        }
    }
#endif // BUILD_AFS_DAEMON

    ///
    /// Client/server imitating a POSIX interface
    ///

    /// Common functions to server and client

    namespace server {
        namespace detail {
            enum class SocketAction : int {
                Connect,
                /// Package description:
                /// - request_action:uint32 = Connect
                ///
                /// Answer:
                /// - port_length:uint32
                /// - port:char[port_length]
            };

            inline const char *get_socket_path() {
                const char *l = std::getenv("AFS_SOCKET");
                if (l) return l;
                return "/tmp/anarchofs.sock";
            }

            inline bool read_from_socket(int socket, void *v, std::size_t size) {
#ifdef BUILD_AFS_DAEMON
                /// Incorrect, tracker cannot be used outside the MPI thread
                anarchofs::detail::tracker t_("read_from_socket");
#endif
                std::size_t total_read = 0;
                while (total_read < size) {
                    ssize_t count =
                        ::read(socket, (void *)((char *)v + total_read), size - total_read);
                    if (count <= 0) return false;
                    total_read += (std::size_t)count;
                }
                return true;
            }

            inline bool write_into_socket(int socket, const void *v, std::size_t size) {
#ifdef BUILD_AFS_DAEMON
                /// Incorrect, tracker cannot be used outside the MPI thread
                anarchofs::detail::tracker t_("write_into_socket");
#endif
                std::size_t total_written = 0;
                while (total_written < size) {
                    ssize_t count = ::write(socket, (void *)((const char *)v + total_written),
                                            size - total_written);
                    if (count <= 0) return false;
                    total_written += (std::size_t)count;
                }
                return true;
            }
        }
    }

#ifdef BUILD_AFS_DAEMON
    namespace server {
        namespace detail {
            /// Return the list of opened file ids associated to a socket

            inline std::unordered_map<int, std::set<FileId>> &get_opened_files_by_sockets() {
                static std::unordered_map<int, std::set<FileId>> file_ids{16};
                return file_ids;
            }

            /// Start tracking opened files by a socket

            inline void start_tracking_files(int socket) {
                get_opened_files_by_sockets().emplace(std::make_pair(socket, std::set<FileId>{}));
            }

            /// Close all opened files associated to a socket

            inline void close_all_files(int socket) {
                for (const auto &file_id : get_opened_files_by_sockets().at(socket))
                    close(file_id, [=](bool) {});
                get_opened_files_by_sockets().erase(socket);
            }

            inline bool process_socket_action(int socket) {
                using namespace anarchofs::detail;

                // Read the request_action
                std::uint32_t action_buffer;
                if (!read_from_socket(socket, &action_buffer, sizeof(action_buffer))) {
                    warning("process_socket_action: error reading action");
                    return false;
                }
                SocketAction action = (SocketAction)action_buffer;

                // Process action
                switch (action) {
                case SocketAction::Connect: {
                    // Write port
                    std::vector<char> answer(sizeof(std::uint32_t) + get_port_name().size());
                    write_as_chars(std::uint32_t(get_port_name().size()), answer.data());
                    std::copy(get_port_name().begin(), get_port_name().end(),
                              answer.data() + sizeof(std::uint32_t));
                    if (!write_into_socket(socket, (const void *)answer.data(), answer.size())) {
                        warning("process_socket_action: error writing on socket\n");
                        return false;
                    }
                    get_func_buffer().push_back([=]() {
                        MPI_Comm new_comm;
                        check_mpi(MPI_Comm_accept(get_port_name().c_str(), MPI_INFO_NULL, 0,
                                                  MPI_COMM_SELF, &new_comm));
                        get_clients().push_back(new_comm);
                    });
                    break;
                }

                default: return false;
                }
                return true;
            }

            inline std::thread &get_socket_thread() {
                static std::thread th;
                return th;
            }

            inline bool &get_finalize_socket_thread() {
                static bool b = false;
                return b;
            }

            inline bool &is_socket_initialized() {
                static bool b = false;
                return b;
            }

            inline void socket_loop() {
                try {
                    anarchofs::detail::log("socket thread is active, baby!\n");

                    struct sockaddr_un addr;
                    memset(&addr, 0, sizeof(struct sockaddr_un));

                    // Copy the socket path
                    const char *socket_path = get_socket_path();
                    anarchofs::detail::log("server listening to socket %s\n", socket_path);
                    if (strnlen(socket_path, sizeof(addr.sun_path)) >= sizeof(addr.sun_path))
                        throw std::runtime_error("socket path is too long");
                    strcpy(addr.sun_path, socket_path);

                    // Make sure that path isn't being used
                    if (remove(socket_path) == -1 && errno != ENOENT)
                        throw std::runtime_error("error removing the socket path");

                    // Create the socket
                    int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
                    if (socket_fd == -1) throw std::runtime_error("could not create socket");

                    addr.sun_family = AF_UNIX;
                    if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) == -1)
                        throw std::runtime_error("could not bind socket");

                    // Set the maximum queue size to 5
                    if (listen(socket_fd, 5) == -1)
                        throw std::runtime_error("could not set maximum queue size");

                    // Mark socket as active
                    is_socket_initialized() = true;

                    std::vector<int> client_sockets;
                    while (!get_finalize_socket_thread()) {
                        fd_set readfds;

                        // Clear the socket set
                        FD_ZERO(&readfds);

                        // Add master socket to set
                        FD_SET(socket_fd, &readfds);
                        int max_sd = socket_fd; ///< max socket id

                        // Add child sockets to set
                        for (const auto &sd : client_sockets) {
                            //if valid socket descriptor then add to read list
                            if (sd > 0) FD_SET(sd, &readfds);

                            //highest file descriptor number, need it for the select function
                            if (sd > max_sd) max_sd = sd;
                        }

                        // Wait for an activity on one of the sockets, indefinitely
                        int activity = select(max_sd + 1, &readfds, NULL, NULL, NULL);
                        if (activity < 0 && errno != EINTR)
                            throw std::runtime_error("select error");

                        // If something happened on the master socket, then its an incoming connection
                        if (FD_ISSET(socket_fd, &readfds)) {
                            int new_socket = accept(socket_fd, NULL, NULL);
                            if (new_socket < 0) throw std::runtime_error("accept error");

                            // Add new socket to array of sockets
                            bool added = false;
                            for (auto &sd : client_sockets) {
                                if (sd == 0) {
                                    sd = new_socket;
                                    added = true;
                                    break;
                                }
                            }
                            if (!added) client_sockets.push_back(new_socket);

                            // Start tracking the files opened this socket
                            start_tracking_files(new_socket);
                        }

                        // Check IO operations on the other sockets
                        for (auto &sd : client_sockets) {
                            if (!FD_ISSET(sd, &readfds)) continue;

                            // Check for incoming messages, otherwise assume closing
                            if (!process_socket_action(sd)) {
                                close_all_files(sd);
                                ::close(sd);
                                sd = 0;
                            }
                        }
                    }
                } catch (const std::exception &e) {
                    anarchofs::detail::show_error(
                        "Error happened in `anarchofs::server::socket_loop`: %s\n", e.what());
                }
            }
        }

        /// Start the processing messages loop

        inline bool start_socket_loop(bool start_new_thread = true) {
            using namespace detail;
            anarchofs::detail::log("requesting starting socket loop\n");
            try {
                if (start_new_thread) {
                    is_socket_initialized() = false;
                    get_finalize_socket_thread() = false;
                    get_socket_thread() = std::thread([=]() { socket_loop(); });
                    while (!is_socket_initialized()) std::this_thread::yield();
                } else {
                    socket_loop();
                }
                return true;
            } catch (const std::exception &e) {
                anarchofs::detail::show_error(
                    "Error happened in `anarchofs::server::start_socket_loop`: %s\n", e.what());
                return false;
            }
        }

        /// Stop the processing messages loop initiated by `start_socket_loop`

        inline bool stop_socket_loop() {
            using namespace detail;
            anarchofs::detail::log("requesting stopping socket loop\n");
            try {
                get_finalize_socket_thread() = true;
                get_socket_thread().join();
                anarchofs::detail::log("socket loop stopped\n");
                return true;
            } catch (const std::exception &e) {
                anarchofs::detail::show_error(
                    "Error happened in `anarchofs::server::stop_socket_loop`: %s\n", e.what());
                return false;
            }
        }
    }
#endif // BUILD_AFS_DAEMON

    namespace client {

        namespace detail {
            inline int get_socket() {
                static int socket = [=]() {
                    struct sockaddr_un addr;
                    memset(&addr, 0, sizeof(struct sockaddr_un));

                    // Copy the socket path
                    const char *socket_path = server::detail::get_socket_path();
                    anarchofs::detail::log("client accessing socket %s\n", socket_path);
                    if (strnlen(socket_path, sizeof(addr.sun_path)) >= sizeof(addr.sun_path))
                        throw std::runtime_error("socket path is too long");
                    strcpy(addr.sun_path, socket_path);

                    // Create the socket
                    int socket_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
                    if (socket_fd == -1) throw std::runtime_error("could not create socket");

                    addr.sun_family = AF_UNIX;
                    if (connect(socket_fd, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) ==
                        -1)
                        throw std::runtime_error("could not bind socket");

                    return socket_fd;
                }();
                return socket;
            }

            inline const MPI_Comm &get_comm() {
                static MPI_Comm comm = [=]() {
                    std::uint32_t action = (std::uint32_t)server::detail::SocketAction::Connect;
                    if (!server::detail::write_into_socket(detail::get_socket(), &action,
                                                           sizeof(action)))
                        throw std::runtime_error("error writing in socket");
                    std::uint32_t port_size = 0;
                    if (!server::detail::read_from_socket(detail::get_socket(), &port_size,
                                                          sizeof(port_size)))
                        throw std::runtime_error("error reading from socket");
                    std::string port(port_size, (char)0);
                    if (!server::detail::read_from_socket(detail::get_socket(), &port[0],
                                                          port_size))
                        throw std::runtime_error("error reading from socket");
                    // Make sure that MPI is initialized
                    int flag;
                    anarchofs::detail::check_mpi(MPI_Initialized(&flag));
                    if (flag == 0)
                        throw std::runtime_error("anarchofs::client::get_comm(): please initialize "
                                                 "MPI before doing any operation");
                    MPI_Comm comm;
                    anarchofs::detail::check_mpi(
                        MPI_Comm_connect(port.c_str(), MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm));
                    return comm;
                }();
                return comm;
            }

            inline int get_dest_rank() {
                int nprocs, this_proc;
                anarchofs::detail::check_mpi(MPI_Comm_rank(get_comm(), &this_proc));
                anarchofs::detail::check_mpi(MPI_Comm_size(get_comm(), &nprocs));
                if (nprocs != 2)
                    throw std::runtime_error(
                        "anarchofs::client:get_dest_rank(): error, invalid communicator");
                return 1 - this_proc;
            }
        }

        /// File handler
        struct File {
            FileId file_id;     ///< file id for remote file
            std::size_t offset; ///< current displacement on remote file
        };

        /// Open a file (for read-only for now)
        /// \param path: path of the file to open
        /// \return: file handler or (null if failed)

        inline File *open(const char *filename) {
            int filename_size = strlen(filename);
            anarchofs::detail::check_mpi(MPI_Send(
                filename, filename_size, MPI_CHAR, detail::get_dest_rank(),
                (int)anarchofs::detail::ClientAction::GlobalOpenRequest, detail::get_comm()));
            FileId file_id;
            anarchofs::detail::check_mpi(
                MPI_Recv(&file_id, sizeof(file_id), MPI_CHAR, detail::get_dest_rank(),
                         (int)anarchofs::detail::ClientAction::GlobalOpenRequest,
                         detail::get_comm(), MPI_STATUS_IGNORE));
            if (file_id != no_file_id)
                return new File{file_id, 0};
            else
                return nullptr;
        }

        /// Change the current cursor of the file handler
        /// \param f: file handler
        /// \param offset: absolute offset of the first element to be read

        inline void seek(File *f, std::size_t offset) { f->offset = offset; }

        /// Write the content of the file into a given buffer
        /// \param f: file handler
        /// \param v: buffer where to write the content
        /// \param n: number of characters to read
        /// \return: number of characters written into the buffer if positive; error code otherwise

        inline std::int64_t read(File *f, char *v, std::size_t n) {
            // Prepare the message
            std::vector<char> buffer(sizeof(FileId) + sizeof(std::size_t) * 2);
            anarchofs::detail::write_as_chars(f->file_id, buffer.data());
            anarchofs::detail::write_as_chars((std::size_t)f->offset,
                                              buffer.data() + sizeof(FileId));
            anarchofs::detail::write_as_chars((std::size_t)n,
                                              buffer.data() + sizeof(FileId) + sizeof(std::size_t));
            anarchofs::detail::check_mpi(
                MPI_Send(buffer.data(), buffer.size(), MPI_CHAR, detail::get_dest_rank(),
                         (int)anarchofs::detail::ClientAction::ReadRequest, detail::get_comm()));

            // Process answer
            MPI_Message msg;
            MPI_Status status;
            anarchofs::detail::check_mpi(MPI_Mprobe(detail::get_dest_rank(), MPI_ANY_TAG,
                                                    detail::get_comm(), &msg, &status));
            int message_size;
            anarchofs::detail::check_mpi(MPI_Get_count(&status, MPI_CHAR, &message_size));
            if (status.MPI_TAG != (int)anarchofs::detail::ClientAction::ReadRequest &&
                message_size != sizeof(std::int64_t))
                throw std::runtime_error("anarchofs::client::read(): error in protocol");
            std::int64_t error_or_size = message_size;
            auto buffer_response =
                status.MPI_TAG == (int)anarchofs::detail::ClientAction::ReadRequest
                    ? v
                    : (char *)&error_or_size;
            anarchofs::detail::check_mpi(
                MPI_Mrecv(buffer_response, message_size, MPI_CHAR, &msg, MPI_STATUS_IGNORE));

            // Advanced the cursor
            if (error_or_size > 0) f->offset += error_or_size;

            return error_or_size;
        }

        /// Close a file hander
        /// \param f: file handler
        /// \return: whether the operation was successful

        inline bool close(File *f) {
            anarchofs::detail::check_mpi(
                MPI_Send(&f->file_id, sizeof(f->file_id), MPI_CHAR, detail::get_dest_rank(),
                         (int)anarchofs::detail::ClientAction::CloseRequest, detail::get_comm()));

            std::uint32_t response = 0;
            anarchofs::detail::check_mpi(
                MPI_Recv(&response, sizeof(response), MPI_CHAR, detail::get_dest_rank(),
                         (int)anarchofs::detail::ClientAction::CloseRequest, detail::get_comm(),
                         MPI_STATUS_IGNORE));

            delete f;

            return response == 0;
        }
    }
}
