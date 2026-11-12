# Python Memory Management & Garbage Collection

- **Reference Counting**: Primary mechanism. Each object tracks how many references point to it. When count reaches 0, memory is deallocated immediately.
- **Cyclic Garbage Collector**: Handles reference cycles using a generational collection algorithm (Generation 0, 1, 2).
