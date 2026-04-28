fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") {
        // `@loader_path` resolves to the dir containing the loaded
        // library — i.e. wherever delocate places this .dylib inside
        // the wheel. `@loader_path/deps` is the build-time location
        // (`target/<triple>/release/deps/libduckdb.dylib`) where
        // libduckdb-sys parks the downloaded dylib, so delocate can
        // see and bundle it before relocation.
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/deps");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/deps");
    }
}
