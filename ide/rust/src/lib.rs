//! Mímir synchronous native client. Loaded by `mimir --rust`; std-only, no Cargo download.
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::ops::Index;
use std::os::fd::FromRawFd;
use std::os::unix::net::UnixStream;
use std::sync::Mutex;

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Clone)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}
impl Default for Value {
    fn default() -> Self {
        Self::Null
    }
}
impl Value {
    pub fn object(entries: &[(&str, Value)]) -> Self {
        Self::Object(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }
    pub fn array(values: &[Value]) -> Self {
        Self::Array(values.to_vec())
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        if let Self::Array(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        if let Self::Object(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Self::Number(v) = self {
            v.parse().ok()
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        if let Self::Number(v) = self {
            v.parse().ok()
        } else {
            None
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        if let Self::Number(v) = self {
            v.parse().ok()
        } else {
            None
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    pub fn parse(s: &str) -> Result<Self> {
        let mut p = Parser {
            data: s.as_bytes(),
            pos: 0,
        };
        let value = p.value(0)?;
        p.space();
        if p.pos != p.data.len() {
            return Err(Error("trailing JSON data".into()));
        }
        Ok(value)
    }
    pub fn stringify(&self) -> Result<String> {
        let mut out = String::new();
        self.encode(&mut out, 0)?;
        Ok(out)
    }
    fn encode(&self, out: &mut String, depth: usize) -> Result<()> {
        if depth > 64 {
            return Err(Error("Mimir value nesting exceeds 64".into()));
        }
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            Self::Number(v) => {
                let parsed = Value::parse(v)?;
                if !matches!(parsed, Self::Number(_)) {
                    return Err(Error("invalid number".into()));
                }
                out.push_str(v);
            }
            Self::String(s) => quote(s, out),
            Self::Array(v) => {
                out.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        out.push(',')
                    }
                    x.encode(out, depth + 1)?;
                }
                out.push(']');
            }
            Self::Object(v) => {
                out.push('{');
                for (i, (k, x)) in v.iter().enumerate() {
                    if i > 0 {
                        out.push(',')
                    }
                    quote(k, out);
                    out.push(':');
                    x.encode(out, depth + 1)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
    /// Invoke a native closure returned by a builder. Errors are propagated immediately.
    pub fn call(&self, args: &[Value]) -> Result<Value> {
        collapse(self.results(args)?)
    }
    /// Preserve Lua's full return tuple, including nil/error and false/error.
    pub fn results(&self, args: &[Value]) -> Result<Vec<Value>> {
        with_client(|client| {
            if self["$mimir_session"] != client.initial["session"] {
                return Err(Error("foreign native function".into()));
            }
            let id = self["$mimir_function"]
                .as_i64()
                .ok_or_else(|| Error("value is not a native function".into()))?;
            client.invoke(id, args)
        })
    }
}
const NULL: Value = Value::Null;
impl Index<&str> for Value {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.as_object().and_then(|v| v.get(key)).unwrap_or(&NULL)
    }
}
impl Index<usize> for Value {
    type Output = Value;
    fn index(&self, key: usize) -> &Value {
        self.as_array().and_then(|v| v.get(key)).unwrap_or(&NULL)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::String(v.into())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
macro_rules! integers {($($t:ty),*)=>{$(impl From<$t> for Value {fn from(v:$t)->Self {Self::Number(v.to_string())}})*}}
integers!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Self::Number(if v.is_finite() {
            format!("{:?}", v)
        } else {
            "invalid non-finite number".into()
        })
    }
}
impl From<f32> for Value {
    fn from(v: f32) -> Self {
        (v as f64).into()
    }
}
impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Self::Array(v.into_iter().map(Into::into).collect())
    }
}
fn quote(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
}
impl Parser<'_> {
    fn space(&mut self) {
        while self.pos < self.data.len()
            && matches!(self.data[self.pos], b' ' | b'\n' | b'\r' | b'\t')
        {
            self.pos += 1
        }
    }
    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| Error("truncated JSON".into()))?;
        self.pos += 1;
        Ok(b)
    }
    fn expect(&mut self, b: u8) -> Result<()> {
        if self.byte()? == b {
            Ok(())
        } else {
            Err(Error("invalid JSON delimiter".into()))
        }
    }
    fn hex(&mut self) -> Result<u32> {
        let mut n = 0;
        for _ in 0..4 {
            n = n * 16
                + (self.byte()? as char)
                    .to_digit(16)
                    .ok_or_else(|| Error("invalid unicode escape".into()))?;
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            match self.byte()? {
                b'"' => return String::from_utf8(out).map_err(|e| Error(e.to_string())),
                b'\\' => match self.byte()? {
                    b'"' => out.push(b'"'),
                    b'\\' => out.push(b'\\'),
                    b'/' => out.push(b'/'),
                    b'b' => out.push(8),
                    b'f' => out.push(12),
                    b'n' => out.push(10),
                    b'r' => out.push(13),
                    b't' => out.push(9),
                    b'u' => {
                        let mut n = self.hex()?;
                        if (0xd800..=0xdbff).contains(&n) {
                            self.expect(b'\\')?;
                            self.expect(b'u')?;
                            let lo = self.hex()?;
                            if !(0xdc00..=0xdfff).contains(&lo) {
                                return Err(Error("invalid surrogate".into()));
                            }
                            n = 0x10000 + ((n - 0xd800) << 10) + (lo - 0xdc00);
                        }
                        let c = char::from_u32(n)
                            .ok_or_else(|| Error("invalid unicode scalar".into()))?;
                        out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                    }
                    _ => return Err(Error("invalid string escape".into())),
                },
                b if b < 32 => return Err(Error("control byte in JSON string".into())),
                b => out.push(b),
            }
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 64 {
            return Err(Error("Mimir value nesting exceeds 64".into()));
        }
        self.space();
        match self.data.get(self.pos).copied() {
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'[') => {
                self.pos += 1;
                let mut v = Vec::new();
                self.space();
                if self.data.get(self.pos) == Some(&b']') {
                    self.pos += 1;
                    return Ok(Value::Array(v));
                }
                loop {
                    v.push(self.value(depth + 1)?);
                    self.space();
                    match self.byte()? {
                        b']' => break,
                        b',' => (),
                        _ => return Err(Error("invalid array".into())),
                    }
                }
                Ok(Value::Array(v))
            }
            Some(b'{') => {
                self.pos += 1;
                let mut v = BTreeMap::new();
                self.space();
                if self.data.get(self.pos) == Some(&b'}') {
                    self.pos += 1;
                    return Ok(Value::Object(v));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    self.space();
                    self.expect(b':')?;
                    v.insert(key, self.value(depth + 1)?);
                    self.space();
                    match self.byte()? {
                        b'}' => break,
                        b',' => (),
                        _ => return Err(Error("invalid object".into())),
                    }
                }
                Ok(Value::Object(v))
            }
            Some(b'n' | b't' | b'f') => {
                for (word, v) in [
                    ("null", Value::Null),
                    ("true", Value::Bool(true)),
                    ("false", Value::Bool(false)),
                ] {
                    if self.data[self.pos..].starts_with(word.as_bytes()) {
                        self.pos += word.len();
                        return Ok(v);
                    }
                }
                Err(Error("invalid literal".into()))
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.pos;
                if self.data[self.pos] == b'-' {
                    self.pos += 1
                }
                match self.byte()? {
                    b'0' => (),
                    b'1'..=b'9' => {
                        while matches!(self.data.get(self.pos), Some(b'0'..=b'9')) {
                            self.pos += 1
                        }
                    }
                    _ => return Err(Error("invalid number".into())),
                }
                for marker in [b'.', b'e'] {
                    let next = self.data.get(self.pos).copied();
                    if next == Some(marker) || (marker == b'e' && next == Some(b'E')) {
                        self.pos += 1;
                        if marker == b'e' && matches!(self.data.get(self.pos), Some(b'+' | b'-')) {
                            self.pos += 1
                        }
                        let digits = self.pos;
                        while matches!(self.data.get(self.pos), Some(b'0'..=b'9')) {
                            self.pos += 1
                        }
                        if digits == self.pos {
                            return Err(Error("invalid number".into()));
                        }
                    }
                }
                let s = std::str::from_utf8(&self.data[start..self.pos]).unwrap();
                let n = s.parse::<f64>().map_err(|e| Error(e.to_string()))?;
                if !n.is_finite() {
                    return Err(Error("non-finite number".into()));
                }
                Ok(Value::Number(s.into()))
            }
            _ => Err(Error("invalid JSON value".into())),
        }
    }
}
struct Client {
    input: BufReader<UnixStream>,
    output: UnixStream,
    initial: Value,
}
impl Client {
    fn receive(input: &mut BufReader<UnixStream>) -> Result<Value> {
        let mut s = String::new();
        if input.read_line(&mut s)? == 0 {
            return Err(Error("Mimir native bridge disconnected".into()));
        }
        Value::parse(&s)
    }
    fn connect() -> Result<Self> {
        // Descriptor 3 is supplied exclusively by the Mimir process.
        if std::env::var_os("MIMIR_NATIVE_BRIDGE").is_none() {
            return Err(Error("run this client with mimir --rust".into()));
        }
        let output = unsafe { UnixStream::from_raw_fd(3) };
        let mut input = BufReader::new(output.try_clone()?);
        let initial = Self::receive(&mut input)?;
        Ok(Self {
            input,
            output,
            initial,
        })
    }
    fn invoke(&mut self, id: i64, args: &[Value]) -> Result<Vec<Value>> {
        let request =
            Value::object(&[("function", id.into()), ("args", Value::array(args))]).stringify()?;
        self.output.write_all(request.as_bytes())?;
        self.output.write_all(b"\n")?;
        let response = Self::receive(&mut self.input)?;
        if let Some(error) = response["error"].as_str() {
            return Err(Error(error.into()));
        }
        response["results"]
            .as_array()
            .map(|v| v.to_vec())
            .ok_or_else(|| Error("invalid native reply".into()))
    }
}
// One channel owner per process; serialize request/reply pairs across Rust threads.
static CLIENT: Mutex<Option<Client>> = Mutex::new(None);
fn with_client<T>(operation: impl FnOnce(&mut Client) -> Result<T>) -> Result<T> {
    let mut slot = CLIENT
        .lock()
        .map_err(|_| Error("native bridge lock poisoned".into()))?;
    operation(client(&mut slot)?)
}
fn client(slot: &mut Option<Client>) -> Result<&mut Client> {
    if slot.is_none() {
        *slot = Some(Client::connect()?)
    }
    Ok(slot.as_mut().unwrap())
}
fn collapse(mut values: Vec<Value>) -> Result<Value> {
    if values.len() >= 2 && (values[0].is_null() || values[0] == Value::Bool(false)) {
        if let Some(s) = values[1].as_str() {
            return Err(Error(s.into()));
        }
    }
    Ok(match values.len() {
        0 => Value::Null,
        1 => values.remove(0),
        _ => Value::Array(values),
    })
}
/// Complete namespace snapshot, including arg, CONF, CONF_PATH, CONF_DIR and ROOTWORK.
pub fn globals() -> Result<Value> {
    with_client(|client| Ok(client.initial["globals"].clone()))
}
/// Call any reference API by dotted name, without a duplicated function catalogue.
pub fn call(path: &str, args: &[Value]) -> Result<Value> {
    collapse(results(path, args)?)
}
pub fn results(path: &str, args: &[Value]) -> Result<Vec<Value>> {
    with_client(|client| {
        // Resolve in place: ordinary calls must not copy the whole API manifest.
        let globals = &client.initial["globals"];
        let name = path.strip_prefix("Mimir.").unwrap_or(path);
        let mut value = if globals["Mimir"][name.split('.').next().unwrap_or("")].is_null() {
            globals
        } else {
            &globals["Mimir"]
        };
        for part in name.split('.') {
            value = &value[part];
        }
        if value["$mimir_session"] != client.initial["session"] {
            return Err(Error("unknown native function".into()));
        }
        let id = value["$mimir_function"]
            .as_i64()
            .ok_or_else(|| Error("value is not a native function".into()))?;
        client.invoke(id, args)
    })
}
/// Preserve float type for integral-valued float buffers, as in Lua's number arrays.
pub fn floats(values: &[f32]) -> Result<Value> {
    let session = with_client(|client| Ok::<_, Error>(client.initial["session"].clone()))?;
    Ok(Value::object(&[
        ("$mimir_session", session),
        ("$mimir_array_kind", "float".into()),
        ("data", values.to_vec().into()),
    ]))
}
pub fn arg() -> Result<Value> {
    Ok(globals()?["arg"].clone())
}
pub fn conf() -> Result<Value> {
    Ok(globals()?["CONF"].clone())
}
pub fn conf_path() -> Result<Value> {
    Ok(globals()?["CONF_PATH"].clone())
}
pub fn conf_dir() -> Result<Value> {
    Ok(globals()?["CONF_DIR"].clone())
}

include!("api.rs");
