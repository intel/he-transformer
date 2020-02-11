load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")
load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license (for zlib)

# TODO: build zlib

cmake_external(
    name = "seal",
    lib_source = ":SEAL-3.4.5/native/src",
    make_commands = [
        "make -j",
        "make install",
    ],
    out_include_dir = "include/SEAL-3.4",
    static_libraries = ["libseal-3.4.a"],
)
