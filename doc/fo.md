# fo

Purpose of fo.cpp is to generate build files for different OS from concise dependencies description.

fo.cpp traverses source tree and creates a project for each leaf directory. It includes all .cpp and .h files in the directory as C++ source files and all .cu and .cuh files as CUDA source files. Depending on OS fo.cpp either creates msvc solution or CMakeLists.txt.

Non leaf directories are expected to have no files, only subdirectories. Leaf directories are expected to contain z.cfg file and source files. All source files in the leaf directory are added to the project. For msvc precompiled headers are used, each directory is expected to contain stdafx.h and stadafx.cpp.

## z.cfg files

Example z.cfg file for a library with single dependency on gpt/data:

```
DEP(
  gpt/data
)
LIBRARY()
```

z.cfg file can contain these declarations
* LIBRARY() -  directory is a library, by default directory is an executable
* DEP(some/library  another/library) - list of libraries currect directory depends on
* NOPCH(filename) - disable precompiled headers for single file filename

Depedency on util is added to all dirs except util itself.

## command line

fo is called with two argments. First argument source_path point to the source folder. Second argument sln_dir is folder name to store build files.

Optional argument -nocuda can be used to skip all CUDA dependent directories.

