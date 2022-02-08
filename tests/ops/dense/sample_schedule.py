from tvm import te


def dense_128x128x4(X, W, T_dense, s):
    """
    Sample Schedule for A Dense Layer

    This schedule is used for testing code generation changes.
    """
    T_dense_i, T_dense_j, T_dense_k = tuple(T_dense.op.axis) + tuple(T_dense.op.reduce_axis)
    T_dense_local, = s.cache_write([T_dense], "local")
    T_dense_local_i_c, T_dense_local_j_c, T_dense_local_k = tuple(T_dense_local.op.axis) + tuple(T_dense_local.op.reduce_axis)
    T_dense_local_i_c_o_i, T_dense_local_i_c_i = s[T_dense_local].split(T_dense_local_i_c, factor=2)
    T_dense_local_i_c_o_o_i, T_dense_local_i_c_o_i = s[T_dense_local].split(T_dense_local_i_c_o_i, factor=8)
    T_dense_local_i_c_o_o_o_i, T_dense_local_i_c_o_o_i = s[T_dense_local].split(T_dense_local_i_c_o_o_i, factor=8)
    T_dense_local_i_c_o_o_o_o, T_dense_local_i_c_o_o_o_i = s[T_dense_local].split(T_dense_local_i_c_o_o_o_i, factor=1)
    T_dense_local_j_c_o_i, T_dense_local_j_c_i = s[T_dense_local].split(T_dense_local_j_c, factor=2)
    T_dense_local_j_c_o_o_i, T_dense_local_j_c_o_i = s[T_dense_local].split(T_dense_local_j_c_o_i, factor=1)
    T_dense_local_j_c_o_o_o_i, T_dense_local_j_c_o_o_i = s[T_dense_local].split(T_dense_local_j_c_o_o_i, factor=32)
    T_dense_local_j_c_o_o_o_o, T_dense_local_j_c_o_o_o_i = s[T_dense_local].split(T_dense_local_j_c_o_o_o_i, factor=2)
    T_dense_local_k_o_i, T_dense_local_k_i = s[T_dense_local].split(T_dense_local_k, factor=4)
    T_dense_local_k_o_o, T_dense_local_k_o_i = s[T_dense_local].split(T_dense_local_k_o_i, factor=1)
    s[T_dense_local].reorder(T_dense_local_i_c_o_o_o_o, T_dense_local_j_c_o_o_o_o, T_dense_local_i_c_o_o_o_i, T_dense_local_j_c_o_o_o_i, T_dense_local_i_c_o_o_i, T_dense_local_j_c_o_o_i, T_dense_local_k_o_o, T_dense_local_k_o_i, T_dense_local_i_c_o_i, T_dense_local_j_c_o_i, T_dense_local_k_i, T_dense_local_i_c_i, T_dense_local_j_c_i)
    T_dense_i_o_i, T_dense_i_i = s[T_dense].split(T_dense_i, factor=16)
    T_dense_i_o_o_i, T_dense_i_o_i = s[T_dense].split(T_dense_i_o_i, factor=8)
    T_dense_i_o_o_o, T_dense_i_o_o_i = s[T_dense].split(T_dense_i_o_o_i, factor=1)
    T_dense_j_o_i, T_dense_j_i = s[T_dense].split(T_dense_j, factor=2)
    T_dense_j_o_o_i, T_dense_j_o_i = s[T_dense].split(T_dense_j_o_i, factor=32)
    T_dense_j_o_o_o, T_dense_j_o_o_i = s[T_dense].split(T_dense_j_o_o_i, factor=2)
    s[T_dense].reorder(T_dense_i_o_o_o, T_dense_j_o_o_o, T_dense_i_o_o_i, T_dense_j_o_o_i, T_dense_i_o_i, T_dense_j_o_i, T_dense_i_i, T_dense_j_i)
    s[T_dense_local].compute_at(s[T_dense], T_dense_j_o_i)
    W_shared = s.cache_read(W, "shared", [T_dense_local])
    W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
    s[W_shared].compute_at(s[T_dense_local], T_dense_local_k_o_o)
    X_shared = s.cache_read(X, "shared", [T_dense_local])
    X_shared_ax0, X_shared_ax1 = tuple(X_shared.op.axis)
    s[X_shared].compute_at(s[T_dense_local], T_dense_local_k_o_o)
    T_dense_i_o_o_o_j_o_o_o_fused = s[T_dense].fuse(T_dense_i_o_o_o, T_dense_j_o_o_o)
    s[T_dense].bind(T_dense_i_o_o_o_j_o_o_o_fused, te.thread_axis("blockIdx.x"))
    T_dense_i_o_o_i_j_o_o_i_fused = s[T_dense].fuse(T_dense_i_o_o_i, T_dense_j_o_o_i)
    s[T_dense].bind(T_dense_i_o_o_i_j_o_o_i_fused, te.thread_axis("vthread"))
    T_dense_i_o_i_j_o_i_fused = s[T_dense].fuse(T_dense_i_o_i, T_dense_j_o_i)
    s[T_dense].bind(T_dense_i_o_i_j_o_i_fused, te.thread_axis("threadIdx.x"))
    W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
    W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=2)
    s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
    W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=256)
    s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
    X_shared_ax0_ax1_fused = s[X_shared].fuse(X_shared_ax0, X_shared_ax1)
    X_shared_ax0_ax1_fused_o, X_shared_ax0_ax1_fused_i = s[X_shared].split(X_shared_ax0_ax1_fused, factor=4)
    s[X_shared].vectorize(X_shared_ax0_ax1_fused_i)
    X_shared_ax0_ax1_fused_o_o, X_shared_ax0_ax1_fused_o_i = s[X_shared].split(X_shared_ax0_ax1_fused_o, factor=256)
    s[X_shared].bind(X_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
    s[T_dense_local].pragma(T_dense_local_i_c_o_o_o_o, "auto_unroll_max_step", 512)
    s[T_dense_local].pragma(T_dense_local_i_c_o_o_o_o, "unroll_explicit", True)
