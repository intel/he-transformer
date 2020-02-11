
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def he_transformer_workspace():
    http_archive(
        name = "zlib",
        build_file = "@//bzl:zlib.BUILD",
        sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
        url = "https://github.com/madler/zlib/archive/v1.2.11.tar.gz",
    )

    http_archive(
        name = "seal",
        build_file = "@//bzl:seal.BUILD",
        sha256 = "1badbab7e98a471c0d2a845db0278dd077e2fd1857434f271ef2b82798620f11",
        url = "https://github.com/microsoft/SEAL/archive/v3.4.5.tar.gz",
    )

    http_archive(
        name = "rules_foreign_cc",
        sha256 = "a2e43b2141cddce94999e26de8075031394ac11fb8075de8aa0b8e13905715ed",
        strip_prefix = "rules_foreign_cc-master",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/master.zip",
    )

    http_archive(
        name = "ngraph",
        build_file = "@//bzl:ngraph.BUILD",
        sha256 = "24e723f7ed47e2c0068fb5868d8ca88542948a13c9de063d8108f2b67785b089",
        strip_prefix = "ngraph-0.28.0-rc.1",
        urls = [
            "https://github.com/NervanaSystems/ngraph/archive/v0.28.0-rc.1.tar.gz",
        ],
    )

    http_archive(
        name = "ngraph-bridge",
        sha256 = "3ff4cdb07f49541076a851001f287b5c8917543e70194647a9feaf638a5249b8",
        url = "https://github.com/tensorflow/ngraph-bridge/archive/v0.22.0-rc4.tar.gz",
        build_file = "@//bzl:ngraph-bridge.BUILD",
    )

    http_archive(
        name = "nlohmann_json_lib",
        build_file = "@//bzl:nlohmann_json.BUILD",
        sha256 = "e0b1fc6cc6ca05706cce99118a87aca5248bd9db3113e703023d23f044995c1d",
        strip_prefix = "json-3.5.0",
        urls = [
            "https://github.com/nlohmann/json/archive/v3.5.0.tar.gz",
        ],
    )

    configure_protobuf()

def configure_protobuf():
    http_archive(
        name = "com_google_protobuf",
        url = "https://github.com/protocolbuffers/protobuf/archive/v3.9.0.tar.gz",
        sha256 = "2ee9dcec820352671eb83e081295ba43f7a4157181dad549024d7070d079cf65",
        strip_prefix = "protobuf-3.9.0",
        build_file = clean_dep("//bzl:protobuf.BUILD"),
    )

    http_archive(
        name = "rules_cc",
        sha256 = "29daf0159f0cf552fcff60b49d8bcd4f08f08506d2da6e41b07058ec50cfeaec",
        strip_prefix = "rules_cc-b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e",
        url = "https://github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.tar.gz",
    )

    http_archive(
        name = "rules_java",
        sha256 = "f5a3e477e579231fca27bf202bb0e8fbe4fc6339d63b38ccb87c2760b533d1c3",
        strip_prefix = "rules_java-981f06c3d2bd10225e85209904090eb7b5fb26bd",
        url = "https://github.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz",
    )

    http_archive(
        name = "rules_proto",
        sha256 = "88b0a90433866b44bb4450d4c30bc5738b8c4f9c9ba14e9661deb123f56a833d",
        strip_prefix = "rules_proto-b0cc14be5da05168b01db282fe93bdf17aa2b9f4",
        url = "https://github.com/bazelbuild/rules_proto/archive/b0cc14be5da05168b01db282fe93bdf17aa2b9f4.tar.gz",
    )

    http_archive(
        name = "six_archive",
        url = "https://bitbucket.org/gutworth/six/get/1.10.0.zip",
        sha256 = "016c8313d1fe8eefe706d5c3f88ddc51bd78271ceef0b75e0a9b400b6a8998a9",
        strip_prefix = "gutworth-six-e5218c3f66a2",
        build_file = clean_dep("//bzl:six.BUILD"),
    )
