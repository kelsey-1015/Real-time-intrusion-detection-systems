RAWTRACE_FILE = {'couchdb_normal': ('raw_tracefile/couchdb_v1_6_normal'),
                 'couchdb_attack': ('raw_tracefile/couchdb_attack_ace'),
                 'mongodb_normal': ('raw_tracefile/mongodb_normal'),
                 'mongodb_attack': ('raw_tracefile/mongodb_brute_force_1', 'raw_tracefile/mongodb_brute_force_2'),
                 'ml': ('raw_tracefile/ml1_normal', 'raw_tracefile/ml2_normal', 'raw_tracefile/ml3_normal',
                       'raw_tracefile/ml4_normal', 'raw_tracefile/ml7_normal')}


# distinct_syscall_normal_trace
FEATURE_DICT_1 = {'futex': 0, 'epoll_ctl': 0, 'write': 0, 'accept': 0, 'epoll_wait': 0, 'timerfd_settime': 0, 'read': 0,
                'sched_yield': 0, 'rt_sigtimedwait': 0, 'wait4': 0, 'select': 0, 'mmap': 0, 'munmap': 0, 'writev': 0,
                'recvfrom': 0, 'close': 0, 'fcntl': 0, 'getsockopt': 0, 'setsockopt': 0, 'getpeername': 0, 'getpid': 0,
                'stat': 0, 'access': 0, 'open': 0, 'fstat': 0, 'lseek': 0, 'pread': 0, 'fsync': 0, 'rename': 0,
                'socket': 0, 'connect': 0, 'poll': 0, 'sendto': 0, 'ioctl': 0, 'bind': 0, 'getsockname': 0, 'times': 0,
                'sysinfo': 0}

# dictinct_syscall_normal_trace_attack1
FEATURE_DICT_2 = {'futex': 0, 'epoll_ctl': 0, 'write': 0, 'accept': 0, 'epoll_wait': 0, 'timerfd_settime': 0, 'read': 0,
                  'sched_yield': 0, 'rt_sigtimedwait': 0, 'wait4': 0, 'select': 0, 'mmap': 0, 'munmap': 0, 'writev': 0,
                  'recvfrom': 0, 'close': 0, 'fcntl': 0, 'getsockopt': 0, 'setsockopt': 0, 'getpeername': 0,
                  'getpid': 0, 'stat': 0, 'access': 0, 'open': 0, 'fstat': 0, 'lseek': 0, 'pread': 0, 'fsync': 0,
                  'rename': 0, 'socket': 0, 'connect': 0, 'poll': 0, 'sendto': 0, 'ioctl': 0, 'bind': 0, 'getsockname': 0,
                  'times': 0, 'sysinfo': 0, 'setsid': 0, 'rt_sigprocmask': 0, 'execve': 0, 'brk': 0, 'mprotect': 0,
                  'arch_prctl': 0, 'rt_sigaction': 0, 'geteuid': 0, 'getppid': 0, 'dup': 0, 'clone': 0, 'kill': 0,
                  'exit_group': 0, 'procexit': 0, 'signaldeliver': 0, 'sigreturn': 0, 'umask': 0, 'lstat': 0, 'newfstatat': 0,
                  'unlinkat': 0, 'pipe': 0, 'vfork': 0}

# dictinct_syscall_normal_trace_attack1_v_6
FEATURE_DICT_3 = {'futex': 0, 'epoll_ctl': 0, 'write': 0, 'accept': 0, 'epoll_wait': 0, 'timerfd_settime': 0, 'read': 0,
                  'sched_yield': 0, 'rt_sigtimedwait': 0, 'wait4': 0, 'select': 0, 'mmap': 0, 'munmap': 0, 'writev': 0,
                  'recvfrom': 0, 'close': 0, 'fcntl': 0, 'getsockopt': 0, 'setsockopt': 0, 'getpeername': 0, 'getpid': 0,
                  'stat': 0, 'access': 0, 'open': 0, 'fstat': 0, 'lseek': 0, 'pread': 0, 'fsync': 0, 'rename': 0, 'socket': 0,
                  'connect': 0, 'poll': 0, 'sendto': 0, 'ioctl': 0, 'bind': 0, 'getsockname': 0, 'times': 0, 'sysinfo': 0,
                  'setsid': 0, 'rt_sigprocmask': 0, 'execve': 0, 'brk': 0, 'mprotect': 0, 'arch_prctl': 0, 'rt_sigaction': 0,
                  'geteuid': 0, 'getppid': 0, 'dup': 0, 'clone': 0, 'kill': 0, 'exit_group': 0, 'procexit': 0, 'signaldeliver': 0,
                  'sigreturn': 0, 'umask': 0, 'lstat': 0, 'newfstatat': 0, 'unlinkat': 0, 'pipe': 0, 'vfork': 0, 'openat': 0,
                  'getdents': 0, 'uname': 0, 'statfs': 0}

# dictinct_syscall_normal_mongodb
FEATURE_DICT_4 = {'futex': 0, 'nanosleep': 0, 'open': 0, 'fstat': 0, 'getdents': 0, 'close': 0, 'getrusage': 0, 'read': 0,
                  'epoll_wait': 0, 'accept': 0, 'epoll_ctl': 0, 'ioctl': 0, 'getsockname': 0, 'setsockopt': 0, 'getsockopt': 0,
                  'getpeername': 0, 'write': 0, 'gettid': 0, 'prctl': 0, 'getrlimit': 0, 'clone': 0, 'set_robust_list': 0,
                  'recvmsg': 0, 'sendmsg': 0, 'writev': 0, 'lstat': 0, 'unlink': 0, 'stat': 0, 'pread': 0, 'fdatasync': 0,
                  'pwrite': 0, 'sched_yield': 0, 'rename': 0, 'mmap': 0, 'mprotect': 0, 'brk': 0, 'ftruncate': 0, 'select': 0,
                  'madvise': 0, 'fallocate': 0, 'shutdown': 0, 'exit': 0, 'procexit': 0}


# mongo+couchdb/normal
FEATURE_DICT_5 = {'futex': 0, 'nanosleep': 0, 'open': 0, 'fstat': 0, 'getdents': 0, 'close': 0, 'getrusage': 0, 'read': 0,
                  'epoll_wait': 0, 'accept': 0, 'epoll_ctl': 0, 'ioctl': 0, 'getsockname': 0, 'setsockopt': 0, 'getsockopt': 0,
                  'getpeername': 0, 'write': 0, 'gettid': 0, 'prctl': 0, 'getrlimit': 0, 'clone': 0, 'set_robust_list': 0,
                  'recvmsg': 0, 'sendmsg': 0, 'writev': 0, 'lstat': 0, 'unlink': 0, 'stat': 0, 'pread': 0, 'fdatasync': 0,
                  'pwrite': 0, 'sched_yield': 0, 'rename': 0, 'mmap': 0, 'mprotect': 0, 'brk': 0, 'ftruncate': 0, 'select': 0,
                  'madvise': 0, 'fallocate': 0, 'shutdown': 0, 'exit': 0, 'procexit': 0, 'timerfd_settime': 0, 'rt_sigtimedwait': 0,
                  'wait4': 0, 'munmap': 0, 'recvfrom': 0, 'fcntl': 0, 'getpid': 0, 'access': 0, 'lseek': 0, 'fsync': 0, 'socket': 0,
                  'connect': 0, 'poll': 0, 'sendto': 0, 'bind': 0, 'times': 0, 'sysinfo': 0, 'setsid': 0, 'rt_sigprocmask': 0,
                  'execve': 0, 'arch_prctl': 0, 'rt_sigaction': 0, 'geteuid': 0, 'getppid': 0, 'dup': 0, 'kill': 0, 'exit_group': 0,
                  'signaldeliver': 0, 'sigreturn': 0, 'umask': 0, 'newfstatat': 0, 'unlinkat': 0, 'pipe': 0, 'vfork': 0, 'openat': 0,
                  'uname': 0, 'statfs': 0}
