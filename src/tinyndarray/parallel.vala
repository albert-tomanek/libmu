delegate float RunParallelWithReduce_Op(int idx);

float RunParallelWithReduce(int size, RunParallelWithReduce_Op op, F reduce, float init_v) {
    //  // Decide parallelization parameters
    //  int n_workers = -1, n_batch = -1, batch_size = -1;
    //  GetParallelParams(size, n_workers, n_batch, batch_size);

    //  if (n_workers <= 1) {
        // Single execution
        float v = init_v;
        for (int i = 0; i < size; i++) {
            // Operation with reduction
            v = reduce(v, op(i));
        }
        return v;
    //  } else {
    //      // Parallel execution
    //      std::atomic<int> next_batch(0);
    //      std::vector<std::thread> workers(static_cast<size_t>(n_workers));
    //      std::vector<float> results(workers.size());
    //      for (size_t t = 0; t < workers.size(); t++) {
    //          workers[t] = std::thread([ =, &next_batch, &results ]() noexcept {
    //              int batch_cnt = 0;
    //              float v = init_v;
    //              while ((batch_cnt = next_batch++) < n_batch) {
    //                  for (int i = 0; i < batch_size; i++) {
    //                      const int idx = batch_size * batch_cnt + i;
    //                      if (size <= idx) {
    //                          break;
    //                      }
    //                      // Operation with reduction
    //                      v = reduce(v, op(idx));
    //                  }
    //              }
    //              results[t] = v;
    //          });
    //      }
    //      for (auto&& worker : workers) {
    //          worker.join();
    //      }
    //      return std::accumulate(results.begin(), results.end(), init_v, reduce);
    //  }
}
