fn main() {
    napi_build::setup();

    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") {
        // The .node loader resolves dependencies relative to the
        // `.node` file itself (its `LC_RPATH`/`DT_RUNPATH`).
        // `@loader_path/deps` covers the build-time
        // (`target/<triple>/release/deps/libduckdb.dylib`); the bare
        // `@loader_path` covers the published-package layout where
        // libduckdb sits next to the `.node` inside the per-platform
        // npm subpackage.
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/deps");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/deps");
    }
}
