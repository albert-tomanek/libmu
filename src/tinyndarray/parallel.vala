static int NUM_WORKERS = -1;
static int BATCH_SCALE = 4;

internal void GetParallelParams(int size, out int n_workers, out int n_batch,
                              out int batch_size) {
    // Fetch the number of workers
    n_workers = NUM_WORKERS;
    if (n_workers <= 0) {
        n_workers = Thread.supported() ? (int) GLib.get_num_processors() : 1;
    }
    // Compute batch size and it number
    n_batch = n_workers * BATCH_SCALE;
    batch_size = size / n_batch + ((size % n_batch != 0) ? 1 : 0);
    n_workers = int.min(n_workers, batch_size);
}

delegate void RunParallelFn(int idx);

void RunParallel(int size, RunParallelFn op) {
    // Decide parallelization parameters
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetParallelParams(size, out n_workers, out n_batch, out batch_size);

    if (n_workers <= 1) {
        //  Single execution
        for (int i = 0; i < size; i++) {
            // Operation
            op(i);
        }
    } else {
        // Parallel execution
        int next_batch = 0;
        Thread<void>[] workers = new Thread<void>[n_workers];
        for (int n = 0; n < n_workers; n++) {
            workers[n] = new Thread<void>(null, () => {
                int batch_cnt = 0;
                while ((batch_cnt = AtomicInt.get(ref next_batch)) < n_batch) {
                    AtomicInt.inc(ref next_batch);
                    for (int i = 0; i < batch_size; i++) {
                        int idx = batch_size * batch_cnt + i;
                        if (size <= idx) {
                            break;
                        }
                        // Operation
                        op(idx);
                    }
                }
            });
        }
        foreach (var worker in workers) {
            worker.join();
        }
    }
}

delegate float RunParallelWithReduceFn(int idx);

float RunParallelWithReduce(int size, RunParallelWithReduceFn op, F reduce, float init_v) {
    // Decide parallelization parameters
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetParallelParams(size, out n_workers, out n_batch, out batch_size);

    if (n_workers <= 1) {
        //  Single execution
        float v = init_v;
        for (int i = 0; i < size; i++) {
            // Operation with reduction
            v = reduce(v, op(i));
        }
        return v;
    } else {
        // Parallel execution
        int next_batch = 0;
        Thread<void>[] workers = new Thread<void>[n_workers];
        float[] results = new float[workers.length];
        
        for (int t = 0; t < workers.length; t++) {
            workers[t] = new Thread<void>(null, () => {
                int batch_cnt = 0;
                float v = init_v;
                while ((batch_cnt = AtomicInt.get(ref next_batch)) < n_batch) {
                    AtomicInt.inc(ref next_batch);
                    for (int i = 0; i < batch_size; i++) {
                        int idx = batch_size * batch_cnt + i;
                        if (size <= idx) {
                            break;
                        }
                        // Operation with reduction
                        v = reduce(v, op(idx));
                    }
                }
                results[t] = v;
            });
        }
        foreach (var worker in workers) {
            worker.join();
        }

        float v = init_v;
        foreach (var result in results) {
            // Operation with reduction
            v = reduce(v, result);
        }
        return v;
    }
}
