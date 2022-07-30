A GLib-based library that tries to emulate Python's `numpy`.
The aim here isn't to be as efficient as possible, it's to emulate the usage of Numpy as closely as possible.

Most of the code for array operations was taken & translated from the C++ project [tinyndarray](https://github.com/takiyu/tinyndarray/). `libmu` won't be considered ABI-complete until all of the functions from `tinyndarray` are translated, and so for now the build script creates a static library (`.a` not `.so`) to be compiled into each binary that uses it.

### Why's it called _Mu_?

So this was my thought flow:

Numpy? Maths. Maths? Greek letters! Hmm, which one is cool, typable, and hasn't been used yet... Mu!

