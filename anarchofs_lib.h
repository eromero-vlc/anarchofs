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

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <dirent.h>
#include <functional>
#include <fuse.h>
#include <memory>
#include <mpi.h>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
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

    namespace detail {

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
                const auto &result = buffer[first_element];

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
            static Buffer<Func> buffer(1024);
            return buffer;
        }

        inline void check_mpi(int error) {
            if (error == MPI_SUCCESS) return;

            char s[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(error, s, &len);

#define CHECK_AND_THROW(ERR)                                                                       \
    if (error == ERR) {                                                                            \
        std::stringstream ss;                                                                      \
        ss << "MPI error: " #ERR ": " << std::string(&s[0], &s[0] + len);                          \
        throw std::runtime_error(ss.str());                                                        \
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
#undef CHECK_AND_THROW
        }

        enum class Action : int {
            GetFileStatusRequest,
            /// Package description:
            /// - request_num:uint32
            /// - path:null-ending string

            GetFileStatusAnswer,
            /// Package description:
            /// - request_num:uint32
            /// - type:FileType
            /// - file_size:size_t

            GlobalOpenRequest,
            /// Package description:
            /// - request_num:uint32
            /// - file_id:uint32
            /// - path:null-ending string

            GlobalOpenAnswer,
            /// Package description:
            /// - request_num:uint32
            /// - file_size:size_t

            ReadRequest,
            /// Package description:
            /// - request_num:uint32
            /// - file_id:uint32
            /// - offset:size_t
            /// - size:size_t

            ReadAnswer,
            /// Package description:
            /// - request_num:uint32
            /// - content:char[*]

            CloseRequest,
            /// Package description:
            /// - request_num:uint32
            /// - file_id:uint32

            GetDirectoryListRequest,
            /// Package description:
            /// - request_num:uint32
            /// - path:null-ending string

            GetDirectoryListAnswer
            /// Package description:
            /// request_num:uint32
            /// [
            ///  - type:FileType
            ///  - path:null-ending string
            /// ]*
        };

        struct MPI_RequestBuffer {
            MPI_Request request;                 ///< MPI request handler
            std::shared_ptr<std::string> buffer; ///< buffer associated to the request
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

        inline std::string replace_hack(const char *path) {
            int this_proc = get_proc_id();
            std::string::size_type n = 0;
            std::string path_s(path);
            std::string re("@NPROC");
            std::string this_proc_s = std::to_string(this_proc);
            while ((n = path_s.find(re)) != std::string::npos) {
                path_s.replace(n, re.size(), this_proc_s);
            }
            return path_s;
        }

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

#ifdef ANARCOFS_LOG
        template <typename... Args> void log(const char *s, Args... args) {
            fuse_log(FUSE_LOG_DEBUG, s, args...);
        }
#else
        template <typename... Args> void log(const char *, Args...) {}
#endif
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

            inline void response_file_status_request(int rank, int message_size) {
                std::vector<char> buffer(message_size + 1);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::GetFileStatusRequest, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                const char *path = buffer.data() + sizeof(RequestNum);
                buffer[message_size] = 0; // make path a null-terminate string
                std::string path_hack = replace_hack(path);
                log("getting requesting get_file_status from %d: %s\n", rank, path_hack.c_str());

                std::string response(sizeof(RequestNum) + sizeof(FileType) + sizeof(Offset), 0);
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

                auto &pending_requests = get_pending_mpi_requests();
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer = std::make_shared<std::string>(std::move(response));
                check_mpi(MPI_Isend(pending_request.buffer->data(), pending_request.buffer->size(),
                                    MPI_CHAR, rank, (int)Action::GetFileStatusAnswer,
                                    MPI_COMM_WORLD, &pending_request.request));
            }

            inline void response_file_status_answer(int rank, int message_size) {
                assert(message_size == sizeof(RequestNum) + sizeof(FileType) + sizeof(Offset));
                std::vector<char> buffer(message_size);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::GetFileStatusAnswer, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                FileType file_type = read_from_chars<FileType>(buffer.data() + sizeof(RequestNum));
                Offset file_size =
                    read_from_chars<Offset>(buffer.data() + sizeof(RequestNum) + sizeof(FileType));
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
            auto &pending_requests = get_pending_mpi_requests();
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                std::string this_msg_pattern = msg_pattern;
                set_request_num(first_req + rank, &this_msg_pattern[0]);
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer = std::make_shared<std::string>(std::move(this_msg_pattern));
                check_mpi(MPI_Isend(pending_request.buffer->data(), pending_request.buffer->size(),
                                    MPI_CHAR, rank, (int)Action::GetFileStatusRequest,
                                    MPI_COMM_WORLD, &pending_request.request));
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

            inline void response_get_directory_list_request(int rank, int message_size) {
                std::vector<char> buffer(message_size + 1);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::GetDirectoryListRequest, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                const char *path = buffer.data() + sizeof(RequestNum);
                buffer[message_size] = 0; // make path a null-terminate string
                log("getting requesting get_directory_list from %d: %s\n", rank, path);

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

                auto &pending_requests = get_pending_mpi_requests();
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer =
                    std::make_shared<std::string>(response.begin(), response.end());
                check_mpi(MPI_Isend(pending_request.buffer->data(), pending_request.buffer->size(),
                                    MPI_CHAR, rank, (int)Action::GetDirectoryListAnswer,
                                    MPI_COMM_WORLD, &pending_request.request));
            }

            inline void response_get_directory_list_answer(int rank, int message_size) {
                std::vector<char> buffer(message_size);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::GetDirectoryListAnswer, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                const char *msg_it = buffer.data() + sizeof(RequestNum);
                std::vector<FilenameType> response;
                for (const char *msg_end = buffer.data() + buffer.size(); msg_it != msg_end;) {
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
            auto &pending_requests = get_pending_mpi_requests();
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                std::string this_msg_pattern = msg_pattern;
                set_request_num(first_req + rank, &this_msg_pattern[0]);
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer = std::make_shared<std::string>(std::move(this_msg_pattern));
                check_mpi(MPI_Isend(pending_request.buffer->data(), pending_request.buffer->size(),
                                    MPI_CHAR, rank, (int)Action::GetDirectoryListRequest,
                                    MPI_COMM_WORLD, &pending_request.request));
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
        using FileInfo = std::vector<Offset>; ///< First offset on each node

        using OpenFilesCache = std::unordered_map<FileId, FileInfo>;

        inline OpenFilesCache &get_open_files_cache() {
            static OpenFilesCache open_files_cache(16);
            return open_files_cache;
        }

        struct LocalOpenedFiles {
            /// From path to file handler
            /// From path to counts
            struct HandlerAndCount {
                std::FILE *f;
                unsigned int count;
            };
            std::unordered_map<std::string, HandlerAndCount> from_path_to_handler_and_count;

            /// From file id to path
            /// From file id to file handler
            struct PathAndHandler {
                std::string path;
                std::FILE *f;
            };
            std::unordered_map<FileId, PathAndHandler> from_file_id_to_path_and_handler;

            LocalOpenedFiles()
                : from_path_to_handler_and_count(16), from_file_id_to_path_and_handler(16) {}

            std::FILE *open(const char *path, const FileId &file_id) {
                std::string path_s(path);
                std::FILE *f;
                if (from_path_to_handler_and_count.count(path) == 0) {
                    f = std::fopen(path, "r");
                    if (f == NULL) return NULL;
                    from_path_to_handler_and_count[path_s] = {f, 1};
                } else {
                    f = from_path_to_handler_and_count[path_s].f;
                }
                from_file_id_to_path_and_handler[file_id] = {path_s, f};
                return f;
            }

            std::FILE *get_file_handler(const FileId &file_id) {
                if (from_file_id_to_path_and_handler.count(file_id) == 0) return NULL;
                return from_file_id_to_path_and_handler.at(file_id).f;
            }

            bool close(const FileId &file_id) {
                if (from_file_id_to_path_and_handler.count(file_id) == 0) return false;
                auto path_and_handler = from_file_id_to_path_and_handler.at(file_id);
                auto handler_and_count = from_path_to_handler_and_count.at(path_and_handler.path);
                if (handler_and_count.count == 1) {
                    std::fclose(handler_and_count.f);
                    from_path_to_handler_and_count.erase(path_and_handler.path);
                } else {
                    from_path_to_handler_and_count.at(path_and_handler.path).count--;
                }
                from_file_id_to_path_and_handler.erase(file_id);
                return true;
            }
        };

        inline LocalOpenedFiles &get_local_opened_files() {
            static LocalOpenedFiles local_opened_files{};
            return local_opened_files;
        }

        namespace detail_open_file {
            inline std::mutex &get_open_file_pending_transactions_mutex() {
                static std::mutex m;
                return m;
            }

            inline std::unordered_map<RequestNum, Promise<std::size_t>> &
            get_open_file_pending_transactions() {
                static std::unordered_map<RequestNum, Promise<std::size_t>> pending(16);
                return pending;
            }

            inline Offset get_file_size(std::FILE *f) {
                // Get the current size of the file
                off_t end_of_file;
                if (std::fseek(f, -1, SEEK_END) != 0)
                    throw std::runtime_error("Error setting file position");

                if ((end_of_file = std::ftell(f) + 1) == 0)
                    throw std::runtime_error("Error getting file position");
                log("getting file size %d\n", (int)end_of_file);

                Offset curr_size = end_of_file;
                return curr_size;
            }

            inline void response_global_open_file_request(int rank, int message_size) {
                std::vector<char> buffer(message_size + 1);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::GlobalOpenRequest, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                FileId file_id = read_from_chars<FileId>(buffer.data() + sizeof(RequestNum));
                const char *path = buffer.data() + sizeof(RequestNum) + sizeof(FileId);
                buffer[message_size] = 0; // make path a null-terminate string
                std::string path_hack = replace_hack(path);
                log("getting requesting get_open_file from %d: %s\n", rank, path_hack.c_str());

                std::string response(sizeof(RequestNum) + sizeof(Offset), 0);
                set_request_num(request_num, &response[0]);

                std::FILE *f = get_local_opened_files().open(path_hack.c_str(), file_id);
                Offset file_size_plus_one = 0;
                if (f != NULL) file_size_plus_one = get_file_size(f) + 1;
                write_as_chars(file_size_plus_one, &response[sizeof(RequestNum)]);

                auto &pending_requests = get_pending_mpi_requests();
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer = std::make_shared<std::string>(std::move(response));
                check_mpi(MPI_Isend(pending_request.buffer->data(), pending_request.buffer->size(),
                                    MPI_CHAR, rank, (int)Action::GlobalOpenAnswer, MPI_COMM_WORLD,
                                    &pending_request.request));
            }

            inline void response_global_open_file_answer(int rank, int message_size) {
                std::vector<char> buffer(message_size);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::GlobalOpenAnswer, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                const char *msg_it = buffer.data() + sizeof(RequestNum);
                Offset file_size_plus_one = read_from_chars<Offset>(msg_it);
                std::unique_lock<std::mutex> unique_lock(
                    get_open_file_pending_transactions_mutex());
                get_open_file_pending_transactions().at(request_num).set(file_size_plus_one);
            }
        }
    }

    /// Open a file
    /// \param path: path of the file to open
    /// \return: file handle

    inline FileId get_open_file(const char *path) {
        using namespace detail;
        using namespace detail_open_file;

        log("get_open_file %s\n", path);

        // Mark the requests from this node
        static RequestNum next_req = 0;
        RequestNum first_req = next_req;
        next_req += get_num_procs();

        // Prepare the responses
        {
            std::unique_lock<std::mutex> unique_lock(get_open_file_pending_transactions_mutex());
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank)
                get_open_file_pending_transactions()[first_req + rank] = {};
        }

        // Create an entry
        static FileId next_file_id = 1;
        if (next_file_id == no_file_id) next_file_id++;
        FileId file_id = next_file_id++;

        // Queue the requests
        std::string msg_pattern =
            std::string(sizeof(RequestNum) + sizeof(FileId), 0) + std::string(path);
        write_as_chars(file_id, &msg_pattern[sizeof(RequestNum)]);
        get_func_buffer().push_back([=]() {
            log("send get_open_file requests %s\n", msg_pattern.c_str() + sizeof(RequestNum));
            auto &pending_requests = get_pending_mpi_requests();
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                std::string this_msg_pattern = msg_pattern;
                set_request_num(first_req + rank, &this_msg_pattern[0]);
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer = std::make_shared<std::string>(std::move(this_msg_pattern));
                check_mpi(MPI_Isend(pending_request.buffer->data(), pending_request.buffer->size(),
                                    MPI_CHAR, rank, (int)Action::GlobalOpenRequest, MPI_COMM_WORLD,
                                    &pending_request.request));
            }
        });

        // Wait for the responses
        std::vector<Offset> file_sizes(get_num_procs());
        bool file_exists = false;
        auto &pending_transactions = get_open_file_pending_transactions();
        for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
            Offset file_size_plus_one = pending_transactions.at(first_req + rank).get();
            if (file_size_plus_one > 0) file_exists = true;
            file_sizes[rank] = file_size_plus_one == 0 ? 0 : file_size_plus_one - 1;
            std::unique_lock<std::mutex> unique_lock(get_open_file_pending_transactions_mutex());
            pending_transactions.erase(first_req + rank);
        }

        // Return special code if the file does not exists
        if (!file_exists) {
            next_file_id--;
            return no_file_id;
        }

        // Get the offsets
        std::vector<Offset> offsets(get_num_procs() + 1);
        for (unsigned int rank = 0; rank < get_num_procs(); ++rank)
            offsets[rank + 1] = offsets[rank] + file_sizes[rank];

        get_open_files_cache()[file_id] = offsets;

        return file_id;
    }

    ///
    /// Read from opened file
    ///

    namespace detail {
        namespace detail_read {
            inline std::mutex &get_read_pending_transactions_mutex() {
                static std::mutex m;
                return m;
            }

            struct StringSizeCount {
                char *buffer;
                std::size_t size;
                std::size_t count;
            };

            inline std::unordered_map<RequestNum, Promise<StringSizeCount>> &
            get_read_pending_transactions() {
                static std::unordered_map<RequestNum, Promise<StringSizeCount>> pending(16);
                return pending;
            }

            inline void response_read_request(int rank, int message_size) {
                std::vector<char> buffer(message_size);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::ReadRequest, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                FileId file_id = read_from_chars<FileId>(buffer.data() + sizeof(RequestNum));
                Offset local_offset =
                    read_from_chars<Offset>(buffer.data() + sizeof(RequestNum) + sizeof(FileId));
                Offset local_size = read_from_chars<Offset>(buffer.data() + sizeof(RequestNum) +
                                                            sizeof(FileId) + sizeof(Offset));
                log("getting requesting read from file id %d %d of characters from %d\n",
                    (int)file_id, (int)local_size, (int)local_offset);

                std::string response(sizeof(RequestNum) + local_size, 0);
                set_request_num(request_num, &response[0]);

                std::FILE *f = get_local_opened_files().get_file_handler(file_id);
                Offset count = 0;
                if (std::fseek(f, local_offset, SEEK_SET) == 0) {
                    count = std::fread(&response[sizeof(RequestNum)], 1, local_size, f);
                }

                auto &pending_requests = get_pending_mpi_requests();
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer = std::make_shared<std::string>(std::move(response));
                check_mpi(MPI_Isend(pending_request.buffer->data(), sizeof(RequestNum) + count,
                                    MPI_CHAR, rank, (int)Action::ReadAnswer, MPI_COMM_WORLD,
                                    &pending_request.request));
            }

            inline void response_read_answer(int rank, int message_size) {
                std::vector<char> buffer(message_size);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::ReadAnswer, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                const char *msg_it = buffer.data() + sizeof(RequestNum);
                Offset count = message_size - sizeof(RequestNum);
                std::unique_lock<std::mutex> unique_lock(get_read_pending_transactions_mutex());
                auto v = get_read_pending_transactions().at(request_num).get_value_unsafe();
                std::copy_n(msg_it, count, v.buffer);
                get_read_pending_transactions().at(request_num).set({v.buffer, v.size, count});
            }
        }
    }

    /// Read from an open file
    /// \param file_id: file handler to read from
    /// \param offset: first character to read
    /// \param count: number of characters to read
    /// \param buffer: memory pointer where to write the content
    /// \return: if positive, the number of read characters; otherwise, error code

    inline std::int64_t read(FileId file_id, std::size_t offset, std::size_t count, char *buffer) {
        using namespace detail;
        using namespace detail_read;

        log("read from file_id %d %d characters starting from %d\n", (int)file_id, (int)count,
            (int)offset);

        // Quick answer if the count is zero
        if (count == 0) return 0;

        // Quick answer if the file_id does not exists
        if (get_open_files_cache().count(file_id) == 0) return -1;

        // Mark the requests from this node
        static RequestNum next_req = 0;
        RequestNum first_req = next_req;
        next_req += get_num_procs();

        // Prepare the responses
        std::vector<Offset> local_offsets(get_num_procs());
        std::vector<Offset> local_counts(get_num_procs());
        {
            std::vector<Offset> str_offsets(get_num_procs());
            const auto &offsets = get_open_files_cache().at(file_id);
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                Offset first_element = std::max(offsets[rank], std::min(offsets[rank + 1], offset));
                Offset last_element =
                    std::max(offsets[rank], std::min(offsets[rank + 1], offset + count));
                if (first_element < last_element) {
                    str_offsets[rank] = first_element - offset;
                    local_offsets[rank] = first_element - offsets[rank];
                    local_counts[rank] = last_element - first_element;
                }
            }
            std::unique_lock<std::mutex> unique_lock(get_read_pending_transactions_mutex());
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank)
                if (local_counts[rank] > 0)
                    get_read_pending_transactions()[first_req + rank] =
                        StringSizeCount{buffer + str_offsets[rank], local_counts[rank], Offset(0)};
        }

        // Queue the requests
        get_func_buffer().push_back([=]() {
            log("send read from file_id %d %d characters starting from %d\n", (int)file_id,
                (int)count, (int)offset);
            auto &pending_requests = get_pending_mpi_requests();
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                if (local_counts[rank] == 0) continue;
                std::string this_msg_pattern =
                    std::string(sizeof(RequestNum) + sizeof(FileId) + sizeof(Offset) * 2, 0);
                set_request_num(first_req + rank, &this_msg_pattern[0]);
                write_as_chars(file_id, &this_msg_pattern[sizeof(RequestNum)]);
                write_as_chars(local_offsets[rank],
                               &this_msg_pattern[sizeof(RequestNum) + sizeof(FileId)]);
                write_as_chars(
                    local_counts[rank],
                    &this_msg_pattern[sizeof(RequestNum) + sizeof(FileId) + sizeof(Offset)]);
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer = std::make_shared<std::string>(std::move(this_msg_pattern));
                check_mpi(MPI_Isend(pending_request.buffer->data(), pending_request.buffer->size(),
                                    MPI_CHAR, rank, (int)Action::ReadRequest, MPI_COMM_WORLD,
                                    &pending_request.request));
            }
        });

        // Wait for the responses
        bool incomplete_request = false;
        std::int64_t res = 0;
        auto &pending_transactions = get_read_pending_transactions();
        for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
            if (local_counts[rank] == 0) continue;
            if (incomplete_request) return -EIO;
            auto string_size_count = pending_transactions.at(first_req + rank).get();
            if (string_size_count.count < string_size_count.size) incomplete_request = true;
            res += (std::int64_t)string_size_count.count;
            std::unique_lock<std::mutex> unique_lock(get_read_pending_transactions_mutex());
            pending_transactions.erase(first_req + rank);
        }

        return res;
    }

    ///
    /// Closed opened file
    ///

    namespace detail {
        namespace detail_close {
            inline std::mutex &get_close_pending_transactions_mutex() {
                static std::mutex m;
                return m;
            }

            struct Void {};

            inline std::unordered_map<RequestNum, Promise<Void>> &get_close_pending_transactions() {
                static std::unordered_map<RequestNum, Promise<Void>> pending(16);
                return pending;
            }

            inline void response_close_request(int rank, int message_size) {
                assert(message_size == sizeof(RequestNum) + sizeof(FileId));
                std::vector<char> buffer(message_size);
                MPI_Status status;
                check_mpi(MPI_Recv(buffer.data(), message_size, MPI_CHAR, rank,
                                   (int)Action::CloseRequest, MPI_COMM_WORLD, &status));
                RequestNum request_num = get_request_num(buffer.data());
                (void)request_num;
                FileId file_id = read_from_chars<FileId>(buffer.data() + sizeof(RequestNum));
                log("getting requesting close from file id %d\n", (int)file_id);

                get_local_opened_files().close(file_id);
            }
        }
    }

    /// Close an open file
    /// \param file_id: file handler to close
    /// \return: true if success

    inline bool close(FileId file_id) {
        using namespace detail;
        using namespace detail_close;

        log("close file_id %d\n", file_id);

        // Quick answer if the file_id does not exists
        if (get_open_files_cache().count(file_id) == 0) return false;

        // Mark the requests from this node
        static RequestNum next_req = 0;
        RequestNum first_req = next_req;
        next_req++;

        // Prepare the responses
        {
            std::unique_lock<std::mutex> unique_lock(get_close_pending_transactions_mutex());
            get_close_pending_transactions()[first_req] = {};
        }

        // Queue the requests
        get_func_buffer().push_back([=]() {
            log("send close requests for file id%d\n", file_id);
            const auto &offsets = get_open_files_cache().at(file_id);
            auto &pending_requests = get_pending_mpi_requests();
            for (unsigned int rank = 0; rank < get_num_procs(); ++rank) {
                if (offsets[rank] == offsets[rank + 1]) continue;
                std::string this_msg_pattern = std::string(sizeof(RequestNum) + sizeof(FileId), 0);
                set_request_num(first_req + rank, &this_msg_pattern[0]);
                write_as_chars(file_id, &this_msg_pattern[sizeof(RequestNum)]);
                pending_requests.resize(pending_requests.size() + 1);
                auto &pending_request = pending_requests.back();
                pending_request.buffer = std::make_shared<std::string>(std::move(this_msg_pattern));
                check_mpi(MPI_Isend(pending_request.buffer->data(), pending_request.buffer->size(),
                                    MPI_CHAR, rank, (int)Action::CloseRequest, MPI_COMM_WORLD,
                                    &pending_request.request));
            }
            std::unique_lock<std::mutex> unique_lock(get_close_pending_transactions_mutex());
            get_close_pending_transactions().at(first_req).set({});
        });

        get_close_pending_transactions().at(first_req).get();
        get_open_files_cache().erase(file_id);

        return true;
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

        inline void mpi_loop(int *argc, char **argv[]) {
            bool mpi_is_active = false; // whether mpi is initialized
            try {
                log("thread is active, baby!\n");
                // Initialize MPI
                int provided = 0;
                check_mpi(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided));
                if (provided < MPI_THREAD_FUNNELED)
                    throw std::runtime_error("MPI does not support the required thread level");
                is_mpi_initialized() = true;
                mpi_is_active = true;

                int nprocs, this_proc;
                check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &this_proc));
                check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
                log("nprocs= %d\n", nprocs);
                get_num_procs() = nprocs;
                get_proc_id() = this_proc;

                auto &buffer = get_func_buffer();
                while (!get_finalize_mpi_thread()) {
                    bool something_done = false;

                    // Check pending MPI requests
                    auto &pending_mpi_requests = get_pending_mpi_requests();
                    if (pending_mpi_requests.size() > 0) {
                        log("checking pending MPI requests\n");
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

                    MPI_Status status;
                    int flag;
                    check_mpi(
                        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status));
                    if (flag != 0) {
                        Action action = (Action)status.MPI_TAG;
                        int origin = status.MPI_SOURCE;
                        int message_size;
                        check_mpi(MPI_Get_count(&status, MPI_CHAR, &message_size));
                        log("got a message from %d\n", origin);
                        switch (action) {
                        case Action::GetFileStatusRequest:
                            detail_get_file_status::response_file_status_request(origin,
                                                                                 message_size);
                            break;

                        case Action::GetFileStatusAnswer:
                            detail_get_file_status::response_file_status_answer(origin,
                                                                                message_size);
                            break;

                        case Action::GlobalOpenRequest:
                            detail_open_file::response_global_open_file_request(origin,
                                                                                message_size);
                            break;

                        case Action::GlobalOpenAnswer:
                            detail_open_file::response_global_open_file_answer(origin,
                                                                               message_size);
                            break;

                        case Action::GetDirectoryListRequest:
                            detail_get_directory_list::response_get_directory_list_request(
                                origin, message_size);
                            break;

                        case Action::GetDirectoryListAnswer:
                            detail_get_directory_list::response_get_directory_list_answer(
                                origin, message_size);
                            break;

                        case Action::ReadRequest:
                            detail_read::response_read_request(origin, message_size);
                            break;

                        case Action::ReadAnswer:
                            detail_read::response_read_answer(origin, message_size);
                            break;

                        case Action::CloseRequest:
                            detail_close::response_close_request(origin, message_size);
                            break;

                        default: throw std::runtime_error("unexpected action code"); break;
                        }
                        something_done = true;
                    }

                    while (buffer.size() > 0) {
                        log("pending function to execute\n");
                        buffer.pop_front()();
                        something_done = true;
                    }

                    if (!something_done) std::this_thread::yield();
                }
            } catch (const std::exception &e) {
                log("Error happened in `anarchofs::mpi_loop`: %s\n", e.what());
            }

            // Finalize MPI
            log("Finalizing MPI\n");
            if (mpi_is_active) MPI_Finalize();
            is_mpi_initialized() = false;
        }
    }

    /// Start the processing messages loop
    /// \param argc: number of arguments (required by MPI_Init)
    /// \param argv: list of commandline arguments (required by MPI_Init)

    bool start_mpi_loop(int *argc, char **argv[]) {
        using namespace detail;
        log("requesting starting MPI loop\n");
        try {
            is_mpi_initialized() = false;
            get_finalize_mpi_thread() = false;
            get_mpi_thread() = std::thread([=]() { mpi_loop(argc, argv); });
            while (!is_mpi_initialized())
                ;
            return true;
        } catch (const std::exception &e) {
            log("Error happened in `anarchofs::start_mpi_loop`: %s\n", e.what());
            return false;
        }
    }

    /// Stop the processing messages loop initiated by `start_mpi_loop`

    bool stop_mpi_loop() {
        using namespace detail;
        log("requesting stopping MPI loop\n");
        try {
            get_finalize_mpi_thread() = true;
            get_mpi_thread().join();
            log("MPI loop stopped\n");
            return true;
        } catch (const std::exception &e) {
            log("Error happened in `anarchofs::stop_mpi_loop`: %s\n", e.what());
            return false;
        }
    }
}
