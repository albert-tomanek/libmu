A GLib-based library that tries to emulate Python's `numpy`.
Not MT-safe.
The aim here isn't to be as efficient as possible, it's to emulate the usage of Numpy as closely as possible.

### Why's it called *Mu*?

So here's my thought flow:

Numpy? Maths. Maths? Greek letters! Hmm, which one is cool, typable, and hasn't been used yet... Mu!

### Note

This library will probably never be feature-complete.
Theoretically it wouldn't be hard to adapt a [C# numpy implementaiton like this one](https://github.com/Quansight-Labs/numpy.net) except that it uses C#-specific features like funciton overloading and partial classes.

---

A lot of code was ~~stolen~~ translated from [tinyndarray](https://github.com/takiyu/tinyndarray/)
