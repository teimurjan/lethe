//! `lethe projects {list, add, remove, prune}` — registry.

use anyhow::Result;
use lethe_core::registry;
use serde::Serialize;
use std::path::PathBuf;

use crate::ProjectsCmdExport as Cmd;

#[derive(Serialize)]
struct ProjectJson {
    root: String,
    slug: String,
    registered_at: String,
}

pub fn run(cmd: Cmd) -> Result<i32> {
    match cmd {
        Cmd::List { json_output } => list(json_output),
        Cmd::Add { path } => add(path.as_deref()),
        Cmd::Remove { name } => remove(&name),
        Cmd::Prune => prune(),
    }
}

fn list(json_output: bool) -> Result<i32> {
    let entries = registry::load();
    if json_output {
        let payload: Vec<ProjectJson> = entries
            .iter()
            .map(|e| ProjectJson {
                root: e.root.to_string_lossy().into_owned(),
                slug: e.slug.clone(),
                registered_at: e.registered_at.clone(),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(0);
    }
    if entries.is_empty() {
        println!("(no registered projects)");
        return Ok(0);
    }
    for e in &entries {
        println!("{}\t{}", e.slug, e.root.display());
    }
    Ok(0)
}

fn add(path: Option<&str>) -> Result<i32> {
    let target = match path {
        Some(p) => PathBuf::from(p)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(p)),
        None => std::env::current_dir()?,
    };
    let entry = registry::register(&target)?;
    println!("registered: {}  {}", entry.slug, entry.root.display());
    Ok(0)
}

fn remove(name: &str) -> Result<i32> {
    let removed = registry::unregister(name)?;
    if !removed {
        eprintln!("no registered project matches {name:?}");
        return Ok(1);
    }
    println!("removed: {name}");
    Ok(0)
}

fn prune() -> Result<i32> {
    let kept = registry::prune()?;
    println!("{} project(s) remain", kept.len());
    Ok(0)
}
