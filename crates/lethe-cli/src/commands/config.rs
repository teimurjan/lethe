//! `lethe config get|set` — TOML config rd/wr.

use anyhow::Result;

use crate::paths::resolve;

use super::store_helpers::{coerce_scalar, load_config, save_config};

pub fn run(
    root: Option<&str>,
    action: &str,
    key: Option<&str>,
    value: Option<&str>,
) -> Result<i32> {
    let paths = resolve(root);
    let mut cfg = load_config(&paths.config_path())?;

    match action {
        "get" => {
            let Some(k) = key else {
                let merged = config_to_toml_table(&cfg);
                println!("{}", toml::to_string_pretty(&merged)?);
                return Ok(0);
            };
            // First check known keys, then `extra` overflow.
            let v: Option<toml::Value> = match k {
                "bi_encoder" => Some(toml::Value::String(cfg.bi_encoder.clone())),
                "cross_encoder" => Some(toml::Value::String(cfg.cross_encoder.clone())),
                "top_k" => Some(toml::Value::Integer(cfg.top_k as i64)),
                "n_clusters" => Some(toml::Value::Integer(cfg.n_clusters as i64)),
                "use_rank_gap" => Some(toml::Value::Boolean(cfg.use_rank_gap)),
                _ => cfg.extra.get(k).cloned(),
            };
            let Some(v) = v else {
                println!("(unset) {k}");
                return Ok(1);
            };
            // Mirror Python: print bare value for str/int/bool, repr for others.
            match v {
                toml::Value::String(s) => println!("{s}"),
                toml::Value::Integer(i) => println!("{i}"),
                toml::Value::Float(f) => println!("{f}"),
                toml::Value::Boolean(b) => println!("{b}"),
                other => println!("{}", toml::to_string(&other)?),
            }
            Ok(0)
        }
        "set" => {
            let (Some(k), Some(v)) = (key, value) else {
                eprintln!("usage: lethe config set KEY VALUE");
                return Ok(2);
            };
            let coerced = coerce_scalar(v);
            match (k, &coerced) {
                ("bi_encoder", toml::Value::String(s)) => cfg.bi_encoder.clone_from(s),
                ("cross_encoder", toml::Value::String(s)) => cfg.cross_encoder.clone_from(s),
                ("top_k", toml::Value::Integer(i)) => cfg.top_k = (*i).max(1) as usize,
                ("n_clusters", toml::Value::Integer(i)) => cfg.n_clusters = (*i).max(0) as u32,
                ("use_rank_gap", toml::Value::Boolean(b)) => cfg.use_rank_gap = *b,
                _ => {
                    cfg.extra.insert(k.to_owned(), coerced.clone());
                }
            }
            save_config(&paths.config_path(), &cfg)?;
            println!("{k} = {coerced:?}");
            Ok(0)
        }
        other => {
            eprintln!("unknown config action: {other}");
            Ok(2)
        }
    }
}

fn config_to_toml_table(cfg: &super::store_helpers::CliConfig) -> toml::Table {
    let mut t = toml::Table::new();
    t.insert(
        "bi_encoder".into(),
        toml::Value::String(cfg.bi_encoder.clone()),
    );
    t.insert(
        "cross_encoder".into(),
        toml::Value::String(cfg.cross_encoder.clone()),
    );
    t.insert("top_k".into(), toml::Value::Integer(cfg.top_k as i64));
    t.insert(
        "n_clusters".into(),
        toml::Value::Integer(cfg.n_clusters as i64),
    );
    t.insert(
        "use_rank_gap".into(),
        toml::Value::Boolean(cfg.use_rank_gap),
    );
    for (k, v) in &cfg.extra {
        t.insert(k.clone(), v.clone());
    }
    t
}
