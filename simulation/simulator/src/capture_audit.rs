//! Opt-in capture audit. No ROS/TF lookup is performed at readback time.
//!
//! A ticket moves inside the driver's (Buffer, Vec<SnapshotSync>, ...) tuple.
//! Its packet id names that *copy operation*, not a reusable GPU allocation.
//! FIFO is used by the existing driver for scheduling only, never for GT lookup.
//! Live GPU synchronization has not been validated by the offline tests here.

use crossbeam_channel::{Sender, TrySendError, bounded};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const VERSION: &str = "part2c2.pilot.frame.v1";
const MANIFEST_VERSION: &str = "part2c2.pilot.manifest.v1";
const WRITER_CAPACITY: usize = 4;
const MAX_PENDING: usize = 8;

fn wall_ns() -> Option<u64> {
    SystemTime::now().duration_since(UNIX_EPOCH).ok()
        .and_then(|v| u64::try_from(v.as_nanos()).ok())
}

#[derive(Debug)]
pub struct AuditFrameSnapshot {
    extraction_id: u64,
    camera_entity: u64,
    width: u32,
    height: u32,
    metadata: Value,
}
impl AuditFrameSnapshot {
    fn new(extraction_id:u64,camera_entity:u64,width:u32,height:u32,metadata:Value)->Result<Self,&'static str> {
        if width==0 || height==0 {return Err("empty_image");}
        Ok(Self {extraction_id,camera_entity,width,height,metadata})
    }
}

#[derive(Default)]
struct Counters {
    allocated: AtomicU64, submitted: AtomicU64, saved: AtomicU64,
    rejected: AtomicU64, abandoned: AtomicU64, queue_full: AtomicU64,
    write_failed: AtomicU64, event_overflow: AtomicU64,
}
impl Counters {
    fn json(&self)->Value {
        let get=|v:&AtomicU64|v.load(Ordering::Relaxed);
        json!({"allocated":get(&self.allocated),"submitted":get(&self.submitted),"saved":get(&self.saved),
            "rejected":get(&self.rejected),"abandoned_before_post_noise":get(&self.abandoned),
            "queue_overflow":get(&self.queue_full),"write_failed":get(&self.write_failed),
            "event_queue_overflow":get(&self.event_overflow)})
    }
}
#[derive(Default)]
struct Ledger {
    last_extraction:u64, next_sequence:u64, next_packet:u64,
    pending:BTreeMap<u64,(u64,u64)>,
}
enum WriteJob {
    Frame {ticket:Value,rgb:Vec<u8>,width:u32,height:u32},
    Event(Value),
    #[cfg(test)] PanicForTest,
}
struct SessionInner {
    id:String,
    // The sole Sender is never cloned. Closing it disconnects the receiver even
    // while late GPU tickets still own AuditSession clones.
    tx:Mutex<Option<Sender<WriteJob>>>,
    ledger:Mutex<Ledger>, counters:Arc<Counters>,
    closing:std::sync::atomic::AtomicBool,
    frame_limit:u64,
    trigger_file:Option<PathBuf>,
    request_quota:AtomicU64,
    worker:Mutex<Option<std::thread::JoinHandle<io::Result<Value>>>>,
    completion:Mutex<Option<Result<Value,String>>>,
}
impl SessionInner {
    fn finish(&self)->io::Result<Value> {
        let mut result=self.completion.lock().map_err(|_|io::Error::other("shutdown lock poisoned"))?;
        if let Some(value)=result.as_ref() {return value.clone().map_err(io::Error::other);}
        self.closing.store(true,Ordering::Release);
        // Only shutdown/main-world teardown waits. GPU callbacks use try_lock
        // and try_send and never join or perform filesystem work.
        let sender=self.tx.lock().map_err(|_|io::Error::other("sender lock poisoned"))?.take();
        if let Ok(mut ledger)=self.ledger.lock() {
            self.counters.abandoned.fetch_add(ledger.pending.len() as u64,Ordering::Relaxed);
            ledger.pending.clear(); // Explicitly abandon unfinished readbacks; never reassign them.
        }
        drop(sender); // all already accepted jobs drain before recv returns Disconnected
        let handle=self.worker.lock().map_err(|_|io::Error::other("worker lock poisoned"))?.take();
        let outcome=match handle {
            Some(handle)=>match handle.join() {
                Ok(value)=>value.map_err(|e|e.to_string()),
                Err(_)=>Err("capture writer panicked; no completion certificate".into()),
            },
            None=>Err("no writer join handle".into()),
        };
        *result=Some(outcome.clone());
        outcome.map_err(io::Error::other)
    }
}
impl Drop for SessionInner {
    fn drop(&mut self) {
        // Fallback for ordinary resource teardown, not a guarantee for SIGKILL,
        // SIGINT default termination, abort, power loss, or a hung filesystem.
        if self.worker.get_mut().is_ok_and(|v|v.is_some()) {
            if let Err(error)=self.finish() {eprintln!("capture shutdown failed: {error}");}
        }
    }
}
#[derive(Clone)]
pub struct AuditSession(Arc<SessionInner>);
impl AuditSession {
    pub fn start(parent:&Path)->io::Result<Self> {
        let limit=match std::env::var("DAEDALUS_AUDIT_MAX_FRAMES") {
            Ok(value)=>value.parse::<u64>().map_err(|_|io::Error::other("invalid DAEDALUS_AUDIT_MAX_FRAMES"))?,
            Err(std::env::VarError::NotPresent)=>0,
            Err(error)=>return Err(io::Error::other(error)),
        };
        let mut session=Self::start_with_limit(parent,limit)?;
        if let Some(path)=std::env::var_os("DAEDALUS_AUDIT_TRIGGER_FILE") {
            let inner=Arc::get_mut(&mut session.0).ok_or_else(||io::Error::other("session unexpectedly shared during initialization"))?;
            inner.trigger_file=Some(PathBuf::from(path));inner.request_quota.store(0,Ordering::Release);
        }
        Ok(session)
    }
    fn start_with_limit(parent:&Path,limit:u64)->io::Result<Self> {
        if !parent.exists() {fs::create_dir(parent)?;}
        if !parent.is_dir() {return Err(io::Error::other("audit output is not a directory"));}
        let id=format!("{}-{}",wall_ns().ok_or_else(||io::Error::other("wall clock before epoch"))?,std::process::id());
        let root=parent.join(format!("part2c2_dataset_{id}"));
        fs::create_dir(&root)?;
        for dir in ["images","metadata","events"] {fs::create_dir(root.join(dir))?;}
        let mut readme=create_new(&root.join("DATASET_README.md"))?;
        readme.write_all(b"# Capture audit Pilot v1\n\nOpt-in exact post-noise RGB8, lossless PNG. No resize/crop/overlay.\nSchema: part2c2.pilot.frame.v1 / part2c2.pilot.manifest.v1, NOT the 2A proposal.\nPNG bytes, decoded RGB/BGR, and metadata are SHA-256 checked by the writer.\nManifest complete means all allocated packets became verified PNG/JSON pairs,\nnot that GPU freshness, physical accuracy, or runtime synchronization passed.\nShutdown closes the sole sender, abandons pending GPU tickets, drains and joins.\nPartial files/orphans are retained as failures, never promoted into valid data.\nAbrupt signals/power loss can leave no valid manifest. A partial manifest is invalid.\nNo runtime GPU verification certificate exists. Source/asset/binary hashes require prelaunch binding.\nDAEDALUS_AUDIT_MAX_FRAMES bounds allocated capture sequences (0=unlimited); no renumbering.\n")?;
        let (tx,rx)=bounded(WRITER_CAPACITY);
        let counters=Arc::new(Counters::default());let wc=counters.clone();let wid=id.clone();
        let worker=std::thread::Builder::new().name("capture-audit-writer".into()).spawn(move|| {
            let mut events=0_u64;let mut completed=Vec::new();
            while let Ok(job)=rx.recv() {
                match job {
                    WriteJob::Frame {ticket,rgb,width,height}=> {
                        let uid=ticket["frame_uid"].as_str().unwrap_or("invalid").to_owned();
                        match write_frame(&root,ticket,&rgb,width,height) {
                            Ok(record)=> {wc.saved.fetch_add(1,Ordering::Relaxed);completed.push(record);}
                            Err(error)=> {
                                wc.write_failed.fetch_add(1,Ordering::Relaxed);events+=1;
                                let _=write_json(&root.join(format!("events/{events:012}.json")),
                                    &json!({"frame_uid":uid,"reason":"write_failed","detail":error.to_string()}));
                            }
                        }
                    }
                    WriteJob::Event(value)=> {
                        events+=1;
                        if write_json(&root.join(format!("events/{events:012}.json")),&value).is_err() {
                            wc.write_failed.fetch_add(1,Ordering::Relaxed);
                        }
                    }
                    #[cfg(test)] WriteJob::PanicForTest=>panic!("injected offline writer failure"),
                }
            }
            finalize_manifest(&root,&wid,limit,&wc,completed)
        })?;
        Ok(Self(Arc::new(SessionInner {id,tx:Mutex::new(Some(tx)),ledger:Mutex::new(Ledger::default()),
            counters,closing:std::sync::atomic::AtomicBool::new(false),frame_limit:limit,
            trigger_file:None,request_quota:AtomicU64::new(u64::MAX),
            worker:Mutex::new(Some(worker)),completion:Mutex::new(None)})))
    }
    pub fn shutdown(&self)->io::Result<Value> {self.0.finish()}
    fn refresh_request_quota(&self) {
        let Some(path)=self.0.trigger_file.as_ref() else {return;};
        // MainWorld Update only, never the GPU readback callback. A missing,
        // partial, malformed, oversized or unreadable local file means disarmed.
        let read=(||->io::Result<u64>{
            let mut text=String::new();File::open(path)?.take(65).read_to_string(&mut text)?;
            if text.len()>64 {return Err(io::Error::other("oversized capture request"));}
            text.trim().parse().map_err(io::Error::other)
        })();
        self.0.request_quota.store(read.unwrap_or(0),Ordering::Release);
    }
    fn accepting(&self)->bool {
        let count=self.0.counters.allocated.load(Ordering::Relaxed);
        !self.0.closing.load(Ordering::Acquire) && count<self.0.request_quota.load(Ordering::Acquire)
            && (self.0.frame_limit==0 || count<self.0.frame_limit)
    }
    fn event(&self,packet:Option<u64>,reason:&str) {
        self.record_event(json!({"session_id":self.0.id,"gpu_copy_packet_id":packet,"reason":reason,"wall_ns":wall_ns()}));
    }
    fn record_event(&self,value:Value) {
        if self.0.closing.load(Ordering::Acquire) {return;}
        let sent=self.0.tx.try_lock().ok().and_then(|guard|guard.as_ref().map(|tx|tx.try_send(WriteJob::Event(value)).is_ok())).unwrap_or(false);
        if !sent {
            // Serialize counter updates with shutdown's final pending-ledger pass.
            if let Ok(_ledger)=self.0.ledger.lock() {
                if !self.0.closing.load(Ordering::Acquire) {
                    self.0.counters.event_overflow.fetch_add(1,Ordering::Relaxed);
                }
            }
        }
    }
    fn claim(&self,snapshot:Arc<AuditFrameSnapshot>)->Result<FrameTicket,&'static str> {
        let sender=self.0.tx.try_lock().map_err(|_|"capture_sender_busy")?;
        if sender.is_none() || !self.accepting() {return Err("capture_closed_or_limit_reached");}
        let mut ledger=self.0.ledger.lock().map_err(|_|"ledger_poisoned")?;
        if snapshot.extraction_id<=ledger.last_extraction {return Err("duplicate_or_stale_extraction");}
        ledger.last_extraction=snapshot.extraction_id;ledger.next_sequence+=1;
        let sequence=ledger.next_sequence;self.0.counters.allocated.fetch_add(1,Ordering::Relaxed);
        if ledger.pending.len()>=MAX_PENDING {
            self.0.counters.rejected.fetch_add(1,Ordering::Relaxed);drop(ledger);drop(sender);
            self.record_event(json!({"session_id":self.0.id,"frame_uid":format!("{}-{sequence:012}",self.0.id),
                "capture_sequence":sequence,"extraction_id":snapshot.extraction_id,"gpu_copy_packet_id":null,"reason":"pending_capacity_exceeded"}));
            return Err("pending_capacity_exceeded");
        }
        ledger.next_packet+=1;let packet_id=ledger.next_packet;
        ledger.pending.insert(packet_id,(snapshot.extraction_id,snapshot.camera_entity));
        Ok(FrameTicket {session:self.clone(),snapshot,packet_id,
            frame_uid:format!("{}-{sequence:012}",self.0.id),capture_sequence:sequence,
            request_quota_at_claim:self.0.trigger_file.as_ref().map(|_|self.0.request_quota.load(Ordering::Acquire)),finished:false})
    }
}

/// Non-Clone, private immutable identity; owned Sync -> exact Buffer tuple -> Async.
pub struct FrameTicket {
    session:AuditSession,snapshot:Arc<AuditFrameSnapshot>,packet_id:u64,
    frame_uid:String,capture_sequence:u64,request_quota_at_claim:Option<u64>,finished:bool,
}
impl FrameTicket {
    fn event(&self,reason:&str) {
        self.session.record_event(json!({"session_id":self.session.0.id,"frame_uid":self.frame_uid,
            "capture_sequence":self.capture_sequence,"gpu_copy_packet_id":self.packet_id,
            "extraction_id":self.snapshot.extraction_id,"camera_entity":self.snapshot.camera_entity,"reason":reason,"wall_ns":wall_ns()}));
    }
    fn reject(&mut self,reason:&'static str)->Result<(),&'static str> {
        self.finished=true;
        if self.session.0.closing.load(Ordering::Acquire) {return Err("capture_session_closed");}
        if let Ok(mut ledger)=self.session.0.ledger.lock() {
            if ledger.pending.remove(&self.packet_id).is_some() {
                self.session.0.counters.rejected.fetch_add(1,Ordering::Relaxed);
            }
        }
        self.event(reason);Err(reason)
    }
    pub fn reject_format(&mut self) {let _=self.reject("image_format_not_rgb8");}
    /// Only read/copy the same post-noise RGB slice published by ROS.
    pub fn post_noise_rgb(&mut self,width:u32,height:u32,rgb:&[u8],noise_sigma:f32,noise_seed:u32)->Result<(),&'static str> {
        if self.finished {return Err("duplicate_ticket_completion");}
        if self.session.0.closing.load(Ordering::Acquire) {self.finished=true;return Err("capture_session_closed");}
        if !noise_sigma.is_finite() {return self.reject("nonfinite_noise_parameter");}
        if (width,height)!=(self.snapshot.width,self.snapshot.height) {return self.reject("image_dimensions_mismatch");}
        if width.checked_mul(height).and_then(|n|n.checked_mul(3)).map(|n|n as usize)!=Some(rgb.len()) {
            return self.reject("image_byte_count_mismatch");
        }
        let mut metadata=self.snapshot.metadata.clone();
        metadata["schema_version"]=json!(VERSION);
        metadata["schema_basis"]=json!({"proposal":"Part2C-2A","conformance":"independent Pilot v1; not backwards-compatible", "full_Draft_2020_12_validation":"not_run"});
        metadata["pilot_control"]=json!({"single_step_enabled":self.session.0.trigger_file.is_some(),
            "requested_capture_quota":self.request_quota_at_claim,
            "semantics":"admission quota frozen at ticket claim; NOT copy identity, FIFO matching or timing evidence"});
        metadata["session_id"]=json!(self.session.0.id);metadata["frame_uid"]=json!(self.frame_uid);
        metadata["extraction_id"]=json!(self.snapshot.extraction_id);metadata["capture_sequence"]=json!(self.capture_sequence);
        metadata["gpu_copy_packet_id"]=json!(self.packet_id);
        metadata["readback_completed_wall_ns"]=json!(wall_ns());
        metadata["readback_time_semantics"]=json!("post-noise CPU callback wall clock; NOT exposure or exact GPU map completion time");
        metadata["gpu_map_completed_wall_ns"]=Value::Null;
        if metadata.get("ros_image_header_stamp").is_none() {
            metadata["ros_image_header_stamp"]=Value::Null;
            metadata["missing_reasons"]["ros_image_header_stamp"]=json!("software fixture has no ROS image header");
        }
        metadata["sync_status"]=json!("same_extraction_structural");metadata["runtime_verified_same_packet"]=json!(false);
        metadata["binding"]=json!({"method":"owned SnapshotSync/Async in exact driver Buffer tuple",
            "registered_ROS_copy_source_compared":metadata["camera"]["copy_source_compared"],
            "registered_ROS_copy_source":metadata["camera"]["registered_ros_copy_source"],
            "gpu_buffer_allocation_id":null,"packet_id_semantics":"unique copy operation, not allocation address",
            "fifo_matching":false,"runtime_gpu_preflight":"not_run"});
        metadata["noise"]=json!({"sigma":noise_sigma,"seed":noise_seed,"seed_applied":noise_sigma>0.0,
            "position":"after add_sensor_noise; same ROS RGB slice"});
        metadata["image"]=json!({"width":width,"height":height,"capture_format":"RGB8","storage_format":"PNG_lossless_RGB",
            "detector_decode":"OpenCV BGR8","ros_frame_id":"camera_optical_frame","resize":false,"crop":false,
            "overlay":false,"post_noise":true,"sha256":null,"decoded_BGR_sha256":null});
        metadata["missing_reasons"]["gpu_map_completed_wall_ns"]=json!("driver exposes RGB callback, not exact map timestamp");
        let job=WriteJob::Frame {ticket:metadata,rgb:rgb.to_vec(),width,height};
        // The sole sender mutex serializes accepted jobs against shutdown, without
        // waiting in this callback. No sender escapes this guard.
        let session=self.session.clone();
        let guard=session.0.tx.try_lock().map_err(|_|"capture_sender_busy")?;
        let Some(tx)=guard.as_ref() else {self.finished=true;return Err("capture_session_closed");};
        if session.0.closing.load(Ordering::Acquire) {self.finished=true;return Err("capture_session_closed");}
        let identity=session.0.ledger.lock().ok().and_then(|mut l|l.pending.remove(&self.packet_id));
        if identity!=Some((self.snapshot.extraction_id,self.snapshot.camera_entity)) {
            drop(guard);return self.reject("packet_snapshot_identity_mismatch");
        }
        self.finished=true;
        match tx.try_send(job) {
            Ok(())=> {session.0.counters.submitted.fetch_add(1,Ordering::Relaxed);Ok(())}
            Err(TrySendError::Full(_))=> {
                session.0.counters.queue_full.fetch_add(1,Ordering::Relaxed);
                if tx.try_send(WriteJob::Event(json!({"frame_uid":self.frame_uid,"reason":"writer_queue_full"}))).is_err() {
                    session.0.counters.event_overflow.fetch_add(1,Ordering::Relaxed);
                }
                Err("writer_queue_full")
            }
            Err(TrySendError::Disconnected(_))=> {
                session.0.counters.write_failed.fetch_add(1,Ordering::Relaxed);Err("writer_disconnected")
            }
        }
    }
}
impl Drop for FrameTicket {
    fn drop(&mut self) {
        if !self.finished {
            let removed=if let Ok(mut ledger)=self.session.0.ledger.lock() {
                if ledger.pending.remove(&self.packet_id).is_some() {
                    self.session.0.counters.abandoned.fetch_add(1,Ordering::Relaxed);true
                } else {false}
            } else {false};
            if removed {self.event("gpu_queue_drop_or_readback_abandoned_unclassified");}
        }
    }
}

fn create_new(path:&Path)->io::Result<File> {OpenOptions::new().write(true).create_new(true).open(path)}
fn write_json(path:&Path,value:&Value)->io::Result<()> {
    let mut file=BufWriter::new(create_new(path)?);
    serde_json::to_writer_pretty(&mut file,value)?;file.flush()?;file.get_ref().sync_all()
}

// Standard SHA-256 compression, no new Cargo dependency or external process.
// Tests include NIST vectors and Python hashlib verifies emitted PNG/JSON hashes.
fn sha256(bytes:&[u8])->String {
    const K:[u32;64]=[
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let mut h=[0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let mut data=bytes.to_vec();data.push(0x80);
    while data.len()%64!=56 {data.push(0);}
    data.extend_from_slice(&((bytes.len() as u64).wrapping_mul(8)).to_be_bytes());
    for block in data.chunks_exact(64) {
        let mut w=[0u32;64];
        for (i,chunk) in block.chunks_exact(4).enumerate() {w[i]=u32::from_be_bytes(chunk.try_into().unwrap());}
        for i in 16..64 {
            let s0=w[i-15].rotate_right(7)^w[i-15].rotate_right(18)^(w[i-15]>>3);
            let s1=w[i-2].rotate_right(17)^w[i-2].rotate_right(19)^(w[i-2]>>10);
            w[i]=w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let [mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh]=h;
        for i in 0..64 {
            let s1=e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25);
            let ch=(e&f)^(!e&g);
            let t1=hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0=a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22);
            let maj=(a&b)^(a&c)^(b&c);let t2=s0.wrapping_add(maj);
            hh=g;g=f;f=e;e=d.wrapping_add(t1);d=c;c=b;b=a;a=t1.wrapping_add(t2);
        }
        for (state,v) in h.iter_mut().zip([a,b,c,d,e,f,g,hh]) {*state=state.wrapping_add(v);}
    }
    h.iter().map(|v|format!("{v:08x}")).collect()
}
fn write_frame(root:&Path,mut metadata:Value,rgb:&[u8],width:u32,height:u32)->io::Result<Value> {
    let uid=metadata["frame_uid"].as_str().ok_or_else(||io::Error::other("missing UID"))?.to_owned();
    if uid.is_empty() || !uid.bytes().all(|v|v.is_ascii_alphanumeric() || v==b'-' || v==b'_') {
        return Err(io::Error::other("unsafe UID"));
    }
    if width==0 || height==0 || width.checked_mul(height).and_then(|n|n.checked_mul(3)).map(|n|n as usize)!=Some(rgb.len()) {
        return Err(io::Error::other("invalid RGB dimensions or length"));
    }
    let image_path=format!("images/{uid}.png");let metadata_path=format!("metadata/{uid}.json");
    let mut file=BufWriter::new(create_new(&root.join(&image_path))?);
    image::write_buffer_with_format(&mut file,rgb,width,height,image::ExtendedColorType::Rgb8,image::ImageFormat::Png).map_err(io::Error::other)?;
    file.flush()?;file.get_ref().sync_all()?;drop(file);
    let decoded=image::open(root.join(&image_path)).map_err(io::Error::other)?.to_rgb8();
    if decoded.width()!=width || decoded.height()!=height || decoded.as_raw()!=rgb {
        return Err(io::Error::other("PNG decoded RGB does not match post-noise source"));
    }
    let png_sha=sha256(&fs::read(root.join(&image_path))?);
    let bgr:Vec<_>=rgb.chunks_exact(3).flat_map(|p|[p[2],p[1],p[0]]).collect();
    metadata["schema_version"]=json!(VERSION);
    metadata["image"]["image_path"]=json!(image_path);metadata["image"]["sha256"]=json!(png_sha);
    metadata["image"]["decoded_RGB_sha256"]=json!(sha256(rgb));metadata["image"]["decoded_BGR_sha256"]=json!(sha256(&bgr));
    metadata["storage_integrity"]=json!({"png_decoded_exact_source_RGB":true,"png_file_synced":true,
        "image_hashes_verified":true,"physical_accuracy_verified":false});
    if let Some(reasons)=metadata.get_mut("missing_reasons").and_then(Value::as_object_mut) {reasons.remove("image_hashes");}
    // Exclusive final names, no overwrite/rename races. A failed JSON leaves a
    // recognizable orphan PNG, excluded from manifest and forcing complete=false.
    write_json(&root.join(&metadata_path),&metadata)?;
    let record=json!({"frame_uid":uid,"capture_sequence":metadata["capture_sequence"],
        "image_path":image_path,"metadata_path":metadata_path,"image_sha256":png_sha,
        "metadata_sha256":sha256(&fs::read(root.join(&metadata_path))?),"complete":true});
    verify_frame_pair(root,&record)?;Ok(record)
}
fn verify_frame_pair(root:&Path,record:&Value)->io::Result<()> {
    let image_path=record["image_path"].as_str().ok_or_else(||io::Error::other("missing image path"))?;
    let metadata_path=record["metadata_path"].as_str().ok_or_else(||io::Error::other("missing metadata path"))?;
    let bytes=fs::read(root.join(metadata_path))?;
    if sha256(&bytes)!=record["metadata_sha256"].as_str().unwrap_or("") {return Err(io::Error::other("metadata SHA mismatch"));}
    let metadata:Value=serde_json::from_slice(&bytes)?;
    if metadata["frame_uid"]!=record["frame_uid"] || metadata["schema_version"]!=VERSION {
        return Err(io::Error::other("UID or schema mismatch"));
    }
    let hash=sha256(&fs::read(root.join(image_path))?);
    if metadata["image"]["image_path"]!=image_path || metadata["image"]["sha256"]!=hash || record["image_sha256"]!=hash {
        return Err(io::Error::other("PNG path or SHA mismatch"));
    }
    let rgb=image::open(root.join(image_path)).map_err(io::Error::other)?.to_rgb8().into_raw();
    let bgr:Vec<_>=rgb.chunks_exact(3).flat_map(|p|[p[2],p[1],p[0]]).collect();
    if metadata["image"]["decoded_RGB_sha256"]!=sha256(&rgb) || metadata["image"]["decoded_BGR_sha256"]!=sha256(&bgr) {
        return Err(io::Error::other("decoded pixel SHA mismatch"));
    }
    Ok(())
}
fn finalize_manifest(root:&Path,id:&str,limit:u64,counters:&Counters,mut frames:Vec<Value>)->io::Result<Value> {
    let mut failures=Vec::new();
    for frame in &mut frames {
        if let Err(error)=verify_frame_pair(root,frame) {
            frame["complete"]=json!(false);failures.push(json!({"frame_uid":frame["frame_uid"],"reason":error.to_string()}));
        }
    }
    let image_count=fs::read_dir(root.join("images"))?.filter_map(Result::ok).count();
    let metadata_count=fs::read_dir(root.join("metadata"))?.filter_map(Result::ok).count();
    let c=counters.json();
    let complete=!frames.is_empty() && failures.is_empty() && image_count==frames.len() && metadata_count==frames.len()
        && c["allocated"]==c["saved"] && c["submitted"]==c["saved"] && c["write_failed"]==0 && c["queue_overflow"]==0;
    let result=json!({"schema_version":MANIFEST_VERSION,"frame_schema_version":VERSION,"session_id":id,
        "complete":complete,"complete_semantics":"all allocated packets are persisted, decoded, hash-verified PNG/JSON pairs; NOT runtime GPU or physical validation",
        "frames":frames,"counters":c,"pair_validation_failures":failures,
        "directory_entries":{"images":image_count,"metadata":metadata_count},"max_capture_sequences":limit,
        "allocated_capture_sequence_range":[1,counters.allocated.load(Ordering::Relaxed)],
        "gap_rule":"allocated sequences minus manifest frame capture_sequence; never renumber",
        "event_loss_note":"event_queue_overflow may lose detailed notices; counters and sequence gaps remain",
        "shutdown":{"sender_closed":true,"accepted_queue_drained":true,"worker_completed":true},
        "ready_for_precision_evaluation":false,"runtime_gpu_preflight":"not_run","runtime_verified_same_packet":false,
        "full_schema_validation":"not_run","missing_reasons":{"runtime_gpu_preflight":"future controlled run required", "provenance_hashes":"bind binary/config/assets before launch"}});
    write_json(&root.join("manifest.json"),&result)?;
    let saved:Value=serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    if saved!=result {return Err(io::Error::other("manifest readback mismatch"));}
    Ok(result)
}
fn validate_copy_source<T:PartialEq>(expected:&T,actual:&T)->Result<(),&'static str> {
    if expected==actual {Ok(())}else{Err("registered_ROS_copy_source_mismatch")}
}
fn audit_backend_supported(talos_feature:bool,force_talos:bool)->bool {!(talos_feature && force_talos)}

// Best-fit plane in physical CENTER coordinates; same definition as Part 2B:
// centroid origin; +x = projected POSITION 0->2; +y = projected 0->1
// orthogonalized against +x; +z = +x cross +y. Raw z residuals are retained.
fn canonical_marker(points: [[f64; 3]; 4]) -> Result<([[f64; 4]; 4], [[f64; 3]; 4]), &'static str> {
    if !points.iter().flatten().all(|v|v.is_finite()) { return Err("nonfinite_marker"); }
    let dot = |a:[f64;3],b:[f64;3]| (0..3).map(|i|a[i]*b[i]).sum::<f64>();
    let sub = |a:[f64;3],b:[f64;3]| std::array::from_fn(|i|a[i]-b[i]);
    let norm = |a:[f64;3]| -> Result<[f64;3],&'static str> {
        let n=dot(a,a).sqrt(); if n < 1e-12 { Err("degenerate_marker") }
        else { Ok(a.map(|x|x/n)) }
    };
    let center = std::array::from_fn(|i|points.iter().map(|p|p[i]).sum::<f64>()/4.0);
    let q=points.map(|p|sub(p,center));
    let mut cov:[[f64;3];3]=std::array::from_fn(|i|std::array::from_fn(|j|q.iter().map(|p|p[i]*p[j]).sum()));
    let mut eigen=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
    for _ in 0..32 {
        let (i,j)=[(0,1),(0,2),(1,2)].into_iter().max_by(|&(a,b),&(c,d)|cov[a][b].abs().total_cmp(&cov[c][d].abs())).unwrap();
        if cov[i][j].abs()<1e-24 {break;}
        let angle=0.5*(2.0*cov[i][j]).atan2(cov[j][j]-cov[i][i]);
        let (s,c)=angle.sin_cos();
        let mut rot=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
        rot[i][i]=c;rot[j][j]=c;rot[i][j]=s;rot[j][i]=-s;
        let tmp:[[f64;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|(0..3).map(|k|cov[a][k]*rot[k][b]).sum()));
        cov=std::array::from_fn(|a|std::array::from_fn(|b|(0..3).map(|k|rot[k][a]*tmp[k][b]).sum()));
        eigen=std::array::from_fn(|a|std::array::from_fn(|b|(0..3).map(|k|eigen[a][k]*rot[k][b]).sum()));
    }
    let smallest=(0..3).min_by(|&i,&j|cov[i][i].total_cmp(&cov[j][j])).unwrap();
    let n=std::array::from_fn(|i|eigen[i][smallest]);
    let raw_x=sub(points[2],points[0]);
    let x=norm(std::array::from_fn(|i|raw_x[i]-n[i]*dot(raw_x,n)))?;
    let raw_y=sub(points[1],points[0]);
    let y=norm(std::array::from_fn(|i|raw_y[i]-n[i]*dot(raw_y,n)-x[i]*dot(raw_y,x)))?;
    let z=[x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]];
    let matrix=std::array::from_fn(|i|if i==3 {[0.,0.,0.,1.]} else {[x[i],y[i],z[i],center[i]]});
    Ok((matrix,q.map(|p|[dot(p,x),dot(p,y),dot(p,z)])))
}

fn unique_center<T:Copy>(entities:&[T])->Result<T,&'static str> {
    match entities { [entity]=>Ok(*entity), []=>Err("CENTER_missing_no_fallback"),
        _=>Err("multiple_CENTER_entities") }
}
fn require_marker(points:Option<[[f64;3];4]>)->Result<[[f64;3];4],&'static str> {
    let points=points.ok_or("MARKER_POSITION_missing")?;
    if !points.iter().flatten().all(|v|v.is_finite()) {return Err("nonfinite_marker");}
    Ok(points)
}
fn finite_matrix(matrix:&[[f64;4];4])->Result<(),&'static str> {
    if matrix.iter().flatten().all(|v|v.is_finite()) {Ok(())}else{Err("nonfinite_transform")}
}

#[cfg(feature = "ros2")]
mod integration {
    use super::*;
    use crate::capture::{CaptureCamera,CaptureSource};
    use crate::robomaster::prelude::{Armor,ArmorRoot,ArmorParts,MarkerData};
    use bevy::prelude::*;
    use bevy::math::{DMat4,DVec3};
    use bevy::camera::{RenderTarget,NormalizedRenderTarget};
    use bevy::render::{ExtractSchedule,MainWorld,RenderApp};
    use bevy::render::camera::{ExtractedCamera,TemporalJitter};
    use bevy::render::view::ExtractedView;
    use bevy::render::sync_world::{RenderEntity,MainEntity};
    use bevy::render::render_asset::RenderAssets;
    use bevy::render::texture::GpuImage;

    #[derive(Resource,Default)]
    struct AuditTick(u64);
    #[derive(Resource)]
    struct AuditState {
        session:AuditSession,
        extraction:u64,
        pending:Option<Frozen>,
        configured:Value,
    }
    struct Frozen {
        snapshot:Arc<AuditFrameSnapshot>,
        main_camera:Entity,
        render_camera:Entity,
        target:Handle<Image>,
        transform:Mat4,
        projection:Mat4,
        viewport:UVec4,
    }

    pub struct CaptureAuditPlugin { pub output:PathBuf }
    impl Plugin for CaptureAuditPlugin {
        fn build(&self,app:&mut App) {
            if !audit_backend_supported(cfg!(feature="talos"),std::env::var("DAEDALUS_FORCE_TALOS_CAPTURE").is_ok_and(|v|v=="1")) {
                error!("Audit disabled: forced ROS+Talos capture is unsupported");return;
            }
            let session=match AuditSession::start(&self.output) {
                Ok(value)=>value,
                Err(error)=>{error!("Audit capture disabled: {error}");return;}
            };
            let configured=read_calibration();
            app.insert_resource(AuditShutdown(session.clone()))
                .add_systems(Last,shutdown_on_exit)
                .add_systems(Update,poll_capture_request)
                .init_resource::<AuditTick>().add_systems(FixedLast,|mut tick:ResMut<AuditTick>|tick.0+=1);
            app.sub_app_mut(RenderApp).insert_resource(AuditState {
                session,extraction:0,pending:None,configured,
            }).add_systems(ExtractSchedule,freeze_capture_state);
        }
    }

    #[derive(Resource)]
    struct AuditShutdown(AuditSession);
    fn poll_capture_request(session:Res<AuditShutdown>) {session.0.refresh_request_quota();}
    fn shutdown_on_exit(mut exits:MessageReader<AppExit>,session:Res<AuditShutdown>) {
        if exits.read().next().is_some() {
            match session.0.shutdown() {
                Ok(manifest)=>info!("Capture writer joined: complete={}",manifest["complete"]),
                Err(error)=>error!("Capture writer shutdown failed: {error}"),
            }
        }
    }

    fn read_calibration()->Value {
        let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/simulation.yaml");
        let text=fs::read_to_string(&path).ok();
        let read=|key:&str,count:usize|->Option<Vec<f64>> {
            let rest=text.as_ref()?.split_once(&format!("{key}:"))?.1;
            let list=rest.split_once('[')?.1.split_once(']')?.0;
            let values:Vec<f64>=list.split(',').map(|v|v.trim().parse::<f64>()).collect::<Result<_,_>>().ok()?;
            (values.len()==count && values.iter().all(|v|v.is_finite())).then_some(values)
        };
        let k=read("camera_matrix",9);let d=read("distort_coeffs",5);
        json!({"source":path,"K":k,"D":d,"source_sha256":null,
            "missing_reasons":{"source_sha256":"offline finalizer required",
            "parse_if_null":"source unavailable or unsupported bracket-list format; no fallback constants"}})
    }

    fn rows(m:DMat4)->[[f64;4];4] {
        let cols=m.to_cols_array_2d();std::array::from_fn(|r|std::array::from_fn(|c|cols[c][r]))
    }
    fn matrix(m:DMat4,from:&str,to:&str,entity:Entity,extraction:u64,affine:bool)->Result<Value,&'static str> {
        finite_matrix(&rows(m))?;
        Ok(json!({"matrix":rows(m),"from_frame":from,"to_frame":to,
            "units":if to=="clip" {"projective homogeneous clip; input view metres"}else if affine {"input raw L mesh units; output metres; linear block includes asset scale/unit conversion"}else{"metres (translation); dimensionless rotation block"},
            "storage_order":"nested rows; column vectors; T_A_from_B",
            "source_entity":entity.to_bits(),"source_schedule":"ExtractSchedule/MainWorld",
            "extraction_id":extraction,"same_extraction":true,"fallback":false,
            "kind":if to=="clip" {"projection_not_rigid"}else if affine {"affine_scale_preserved"}else{"rigid"}}))
    }

    fn center_entity(world:&World,root:Entity)->Result<Entity,&'static str> {
        let mut todo=vec![root];let mut result=Vec::new();let mut visited=0;
        while let Some(entity)=todo.pop() {
            visited+=1;if visited>4096 {return Err("hierarchy_limit");}
            if world.get::<Name>(entity).is_some_and(|name|name.as_str().ends_with("CENTER")) {
                result.push(entity);
            }
            if let Some(children)=world.get::<Children>(entity) {todo.extend(children.iter());}
        }
        unique_center(&result)
    }

    fn asset_name(world:&World,mut entity:Entity)->Option<String> {
        for _ in 0..64 {
            if let Some(asset)=world.get::<WorldAssetRoot>(entity) {
                return asset.0.path().map(|p|p.to_string());
            }
            entity=world.get::<ChildOf>(entity)?.parent();
        }
        None
    }

    fn training_target_identity(world:&World,mut entity:Entity)->Option<Value> {
        for _ in 0..64 {
            if let Some(target)=world.get::<crate::training::TrainingTarget>(entity) {
                return Some(json!({"entity":entity.to_bits(),"kind":target.kind,
                    "name":world.get::<Name>(entity).map(|n|n.as_str()),
                    "source":"actual TrainingTarget ancestor in the same MainWorld extraction"}));
            }
            entity=world.get::<ChildOf>(entity)?.parent();
        }
        None
    }

    fn capture_target(world:&World,entity:Entity,root:&ArmorRoot,parts:&ArmorParts,armor:&Armor,
        extraction:u64)->Result<Value,&'static str> {
        let center=center_entity(world,entity)?;
        let marker=parts.marker();
        let bs=world.get::<GlobalTransform>(center).ok_or("CENTER_GlobalTransform_missing")?.to_matrix().as_dmat4();
        let bl=world.get::<GlobalTransform>(marker).ok_or("MARKER_GlobalTransform_missing")?.to_matrix().as_dmat4();
        let local=require_marker(world.get::<MarkerData>(marker).map(|m|m.0.map(|p|p.as_dvec3().to_array())))?;
        if !bs.is_finite() || !bl.is_finite() {return Err("nonfinite_transform");}
        if bs.determinant().abs()<1e-12 || bl.determinant().abs()<1e-12 {return Err("singular_transform");}
        // CENTER is required to be a metric rigid frame; mesh L is allowed scaling.
        let a=bs.x_axis.truncate();let b=bs.y_axis.truncate();let c=bs.z_axis.truncate();
        if [(a.length_squared()-1.).abs(),(b.length_squared()-1.).abs(),(c.length_squared()-1.).abs(),
            a.dot(b).abs(),a.dot(c).abs(),b.dot(c).abs(),(a.dot(b.cross(c))-1.).abs()]
            .iter().any(|v|*v>1e-4) {return Err("CENTER_not_metric_rigid");}
        let sl=bs.inverse()*bl;
        let in_s=local.map(|p|sl.transform_point3(DVec3::from_array(p)).to_array());
        let (so_rows,raw_o)=canonical_marker(in_s)?;
        let so=DMat4::from_cols_array_2d(&std::array::from_fn(|i|std::array::from_fn(|j|so_rows[j][i])));
        let bo=bs*so;
        let plane_o=raw_o.map(|p|[p[0],p[1],0.]);
        let closure=(0..4).map(|i|bl.transform_point3(DVec3::from_array(local[i]))
            .distance(bo.transform_point3(DVec3::from_array(raw_o[i])))).fold(0.0_f64,f64::max);
        if closure>1e-8 {return Err("marker_normalization_closure_failed");}
        Ok(json!({"training_target":training_target_identity(world,entity),
            "armor_id":root.id.as_usize(),"armor_type":format!("{:?}",armor.spec.armor_type()),
            "armor_entity":entity.to_bits(),"armor_label":format!("{:?}",armor.label),
            "CENTER_entity":center.to_bits(),"MARKER_mesh_entity":marker.to_bits(),
            "asset_name":asset_name(world,entity),"asset_sha256":null,
            "missing_reasons":{"asset_sha256":"offline finalizer required", "occlusion":"not measured; hidden helper MARKER is not armor occlusion"},
            "T_B_from_S":matrix(bs,"S_CENTER","B_Bevy_world",center,extraction,false)?,
            "T_S_from_O":matrix(so,"O_normalized_MARKER","S_CENTER",marker,extraction,false)?,
            "T_B_from_O":matrix(bo,"O_normalized_MARKER","B_Bevy_world",marker,extraction,false)?,
            "T_B_from_meshMARKER":matrix(bl,"L_meshMARKER","B_Bevy_world",marker,extraction,true)?,
            "T_S_from_meshMARKER":matrix(sl,"L_meshMARKER","S_CENTER",marker,extraction,true)?,
            "POSITION_indices":[0,1,2,3],"POSITION_L":local,"POSITION_S_m":in_s,
            "POSITION_L_units":"raw runtime mesh units; convert only through T_S_from_meshMARKER",
            "POSITION_indices_source":"MarkerData Mesh::ATTRIBUTE_POSITION elements; not triangle indices",
            "POSITION_O_raw_m":raw_o,"POSITION_O_planar_m":plane_o,
            "geometry_version":"actual extracted mesh; fitted plane eigenbasis part2b convention",
            "O_definition":"centroid; +x projected 0->2; +y projected 0->1 orthogonalized; +z=x cross y",
            "normalization_closure_max_m":closure,
            "visibility":{"marker_local":world.get::<Visibility>(marker).map(|v|format!("{v:?}")),
                "marker_inherited":world.get::<InheritedVisibility>(marker).map(|v|v.get()),
                "marker_any_view":world.get::<ViewVisibility>(marker).map(|v|v.get()),
                "armor_local":world.get::<Visibility>(entity).map(|v|format!("{v:?}")),
                "armor_inherited":world.get::<InheritedVisibility>(entity).map(|v|v.get()),
                "occlusion":null,"actual_capture_view_visibility":null},
            "fallback":false}))
    }

    fn freeze_capture_state(main:Res<MainWorld>,mut state:ResMut<AuditState>) {
        state.extraction+=1;state.pending=None;
        if !state.session.accepting() {return;}
        match freeze(&main,state.extraction,state.configured.clone()) {
            Ok(frozen)=>state.pending=Some(frozen),
            Err(reason)=>{state.session.0.counters.rejected.fetch_add(1,Ordering::Relaxed);
                state.session.record_event(json!({"session_id":state.session.0.id,"extraction_id":state.extraction,
                    "gpu_copy_packet_id":null,"frame_uid":null,"reason":reason,"source_schedule":"ExtractSchedule"}));}
        }
    }

    #[allow(deprecated)] // read-only World traversal; no MainWorld mutation during extraction
    fn freeze(world:&World,extraction:u64,configured:Value)->Result<Frozen,&'static str> {
        let cameras:Vec<_>=world.iter_entities().filter(|e|e.contains::<CaptureCamera>()).collect();
        if cameras.len()!=1 {return Err("CaptureCamera_not_unique");}
        let entity=cameras[0].id();
        let camera=world.get::<Camera>(entity).ok_or("Camera_missing")?;
        if !camera.is_active {return Err("capture_camera_inactive_stale_texture");}
        if world.get::<TemporalJitter>(entity).is_some() {return Err("temporal_jitter_not_supported");}
        let render_camera=world.get::<RenderEntity>(entity).ok_or("RenderEntity_missing")?.id();
        let transform=world.get::<GlobalTransform>(entity).ok_or("camera_transform_missing")?.to_matrix();
        let projection=camera.clip_from_view();
        if !transform.is_finite() || !projection.is_finite() {return Err("nonfinite_camera");}
        if transform.determinant().abs()<1e-12 {return Err("singular_camera");}
        let rect=camera.physical_viewport_rect().ok_or("viewport_missing")?;
        let size=camera.physical_target_size().ok_or("target_size_missing")?;
        if rect.min!=UVec2::ZERO || rect.size()!=size {return Err("partial_viewport_not_supported");}
        let target=world.get_resource::<crate::ros2::capture::RosAuditCopySource>().ok_or("ROS_registered_source_missing")?.handle().clone();
        let expected=RenderTarget::Image(target.clone().into());
        let actual=world.get::<RenderTarget>(entity).ok_or("RenderTarget_missing")?;
        if actual.normalize(None)!=expected.normalize(None) {return Err("capture_target_mismatch");}
        if world.iter_entities().filter(|e|e.get::<Camera>().is_some_and(|c|c.is_active)
            && e.get::<RenderTarget>().is_some_and(|t|t.normalize(None)==expected.normalize(None))).count()!=1 {return Err("multiple_cameras_same_target");}
        let Projection::Perspective(perspective)=world.get::<Projection>(entity).ok_or("projection_missing")?
            else {return Err("non_perspective_camera");};
        let sources:Vec<_>=world.iter_entities().filter(|e|e.contains::<CaptureSource>()).collect();
        let source=if sources.len()==1 {Some(sources[0].id())} else {None};
        let bv=transform.as_dmat4();
        let axes=[bv.x_axis.truncate(),bv.y_axis.truncate(),bv.z_axis.truncate()];
        if axes.iter().any(|v|(v.length_squared()-1.).abs()>1e-4)
            || axes[0].dot(axes[1]).abs()>1e-4 || axes[0].dot(axes[2]).abs()>1e-4
            || axes[1].dot(axes[2]).abs()>1e-4 || (bv.determinant()-1.).abs()>1e-4 {
            return Err("CaptureCamera_not_metric_rigid");
        }
        let q=DMat4::from_diagonal(bevy::math::DVec4::new(1.,-1.,-1.,1.));
        let pc=rows(projection.as_dmat4());
        // Reject non-pinhole/skew/oblique matrices rather than inventing K.
        if pc[3]!=[0.,0.,-1.,0.] || pc[0][1]!=0. || pc[1][0]!=0. || pc[0][3]!=0. || pc[1][3]!=0. {
            return Err("unsupported_projection");
        }
        let k=[[size.x as f64*pc[0][0]/2.,0.,size.x as f64*(1.-pc[0][2])/2.],
            [0.,size.y as f64*pc[1][1]/2.,size.y as f64*(1.+pc[1][2])/2.],[0.,0.,1.]];
        let mut targets=Vec::new();let mut rejected=Vec::new();
        for item in world.iter_entities() {
            if let Some(root)=item.get::<ArmorRoot>() {
                let value=match (item.get::<ArmorParts>(),item.get::<Armor>()) {
                    (Some(parts),Some(armor))=>capture_target(world,item.id(),root,parts,armor,extraction),
                    _=>Err("armor_parts_or_type_missing"),
                };
                match value {Ok(value)=>targets.push(value),Err(reason)=>rejected.push(json!({"armor_id":root.id.as_usize(),"entity":item.id().to_bits(),"reason":reason}))}
            }
        }
        if targets.is_empty() {return Err("no_valid_CENTER_MARKER_target");}
        let training=world.get_resource::<crate::config::SimulationConfig>().map(|c|&c.training);
        let selected=world.get_resource::<crate::scoreboard::Board>().map_or(0,|b|b.0.selected);
        if training.is_some_and(|c|c.enabled) && !targets.iter().any(|t|t["training_target"]["kind"].as_u64()==Some(selected as u64)) {
            return Err("selected_training_target_CENTER_MARKER_not_ready");
        }
        let virtual_time=world.get_resource::<Time<Virtual>>();
        let fixed=world.get_resource::<Time<Fixed>>();
        let ns=|v:std::time::Duration|u64::try_from(v.as_nanos()).ok();
        let metadata=json!({"evidence_kind":"actual_capture","partition":"engineering_diagnostics",
            "scene":{"training_enabled":training.map(|c|c.enabled),"selected_training_target_kind":selected,
                "solo_target":training.map(|c|c.solo_target),
                "source":"same extraction MainWorld SimulationConfig and Board; kind0=static,1=translating,2=rotating"},
            "simulation_tick":world.get_resource::<AuditTick>().map(|t|t.0),
            "simulation_tick_source":"AuditTick increments once per FixedLast since plugin registration; not physics substep or extraction count",
            "capture_timestamp":{"system_time_ns":wall_ns(),
                "virtual_time_ns":virtual_time.and_then(|t|ns(t.elapsed())),
                "fixed_time_ns":fixed.and_then(|t|ns(t.elapsed())),
                "fixed_timestep_ns":fixed.and_then(|t|ns(t.timestep())),
                "fixed_overstep_ns":fixed.and_then(|t|ns(t.overstep())),
                "exposure_time_ns":null,"semantics":"same MainWorld state, not measured exposure"},
            "camera":{"CaptureCamera_entity":entity.to_bits(),"render_entity":render_camera.to_bits(),
                "CaptureSource_entity":source.map(|e|e.to_bits()),"render_target":format!("{actual:?}"),
                "T_B_from_V":matrix(bv,"V_Bevy_view","B_Bevy_world",entity,extraction,false)?,
                "T_V_from_C":matrix(q,"C_OpenCV_optical","V_Bevy_view",entity,extraction,false)?,
                "T_B_from_C":matrix(bv*q,"C_OpenCV_optical","B_Bevy_world",entity,extraction,false)?,
                "CaptureSource_T_B_from_V":source.and_then(|e|world.get::<GlobalTransform>(e).and_then(|t|matrix(t.to_matrix().as_dmat4(),"CaptureSource_local","B_Bevy_world",e,extraction,false).ok())),
                "clip_from_view":matrix(projection.as_dmat4(),"V_Bevy_view","clip",entity,extraction,false)?,
                "viewport":[rect.min.x,rect.min.y,size.x,size.y],"width":size.x,"height":size.y,
                "K_actual_from_projection":k,"calibration_config":configured,
                "D_rendered":[0.,0.,0.,0.,0.],"D_source":"undistorted Perspective renderer; config preserved separately",
                "projection_type":"Perspective","fov_y_rad":perspective.fov,"near_m":perspective.near,
                "far_m":if perspective.far.is_finite(){Some(perspective.far)}else{None},
                "temporal_jitter_present":false,"extracted_view_compared":false,
                "projection_runtime_checked":false},
            "targets":targets,"rejected_targets":rejected,
            "detector":{"status":"not_run","raw_Armor_outputs":null,"reason":"separate offline tool required; detector unchanged"},
            "missing_reasons":{"exposure_time_ns":"GPU exposure time not measured",
                "CaptureSource_if_null":"not exactly one source or transform absent",
                "times_if_null":"time resource absent or nanosecond conversion overflow",
                "visibility_null":"no view-specific occlusion measurement",
                "provenance_hashes":"offline finalizer must bind source/assets/config/binary"}});
        Ok(Frozen {snapshot:Arc::new(AuditFrameSnapshot::new(extraction,entity.to_bits(),size.x,size.y,metadata)?),
            main_camera:entity,render_camera,target,transform,projection,
            viewport:UVec4::new(rect.min.x,rect.min.y,size.x,size.y)})
    }

    pub fn ticket_for_copy(world:&World,registered_source:Option<&Handle<Image>>,ros_sec:i32,ros_nanosec:u32)->Option<FrameTicket> {
        let state=world.get_resource::<AuditState>()?;
        if !state.session.accepting() {return None;}
        let result=(||->Result<FrameTicket,&'static str>{
            if world.get_resource::<crate::ros2::capture::RosCaptureContext>().is_some_and(|c|c.publish_compressed) {
                return Err("ROS_compressed_path_cannot_guarantee_pixel_equivalence");
            }
            if ros_nanosec>=1_000_000_000 {return Err("invalid_ROS_header_nanosec");}
            let source=registered_source.ok_or("ROS_registered_source_uninitialized")?;
            let f=state.pending.as_ref().ok_or("snapshot_missing")?;
            validate_copy_source(&f.target.id(),&source.id())?;
            if f.snapshot.extraction_id!=state.extraction {return Err("stale_extraction");}
            if world.get::<MainEntity>(f.render_camera).map(|e|e.id())!=Some(f.main_camera) {return Err("camera_entity_mismatch");}
            let view=world.get::<ExtractedView>(f.render_camera).ok_or("ExtractedView_missing_stale_texture")?;
            let camera=world.get::<ExtractedCamera>(f.render_camera).ok_or("ExtractedCamera_missing_stale_texture")?;
            if view.world_from_view.to_matrix()!=f.transform || view.clip_from_view!=f.projection
                || view.viewport!=f.viewport || view.clip_from_world.is_some() {
                return Err("ExtractedView_transform_projection_viewport_mismatch");
            }
            if world.get::<TemporalJitter>(f.render_camera).is_some() {return Err("render_temporal_jitter");}
            if camera.target!=Some(NormalizedRenderTarget::Image(f.target.clone().into()))
                || camera.physical_target_size!=Some(UVec2::new(f.snapshot.width,f.snapshot.height)) {
                return Err("ExtractedCamera_target_mismatch");
            }
            let images=world.get_resource::<RenderAssets<GpuImage>>().ok_or("gpu_images_missing")?;
            let gpu=images.get(source).ok_or("gpu_color_texture_missing")?;
            if gpu.texture_descriptor.size.width!=f.snapshot.width || gpu.texture_descriptor.size.height!=f.snapshot.height {
                return Err("gpu_texture_dimensions_mismatch");
            }
            let mut metadata=f.snapshot.metadata.clone();
            metadata["ros_image_header_stamp"]=json!({"sec":ros_sec,"nanosec":ros_nanosec,
                "clock":"ROS SystemTime","semantics":"exact header passed to ROS raw_image; generated in copy handler, not exposure time"});
            metadata["camera"]["copy_source_compared"]=json!(true);
            metadata["camera"]["registered_ros_copy_source"]=json!(format!("{:?}",source.id()));
            metadata["camera"]["extracted_view_compared"]=json!(true);
            metadata["camera"]["ExtractedView_source_schedule"]=json!("RenderGraph after camera_driver; exact comparison with frozen ExtractSchedule state");
            metadata["camera"]["projection_comparison"]=json!("actual ExtractedView clip_from_view equals frozen Camera::clip_from_view; not a GPU pixel test");
            // No auto-promotion to runtime_verified_same_packet. A future controlled
            // preflight must verify rendered pixels/freshness, not merely CPU matrices.
            let snapshot=Arc::new(AuditFrameSnapshot::new(state.extraction,f.main_camera.to_bits(),f.snapshot.width,f.snapshot.height,metadata)?);
            state.session.claim(snapshot)
        })();
        match result {Ok(ticket)=>Some(ticket),Err(reason)=>{state.session.0.counters.rejected.fetch_add(1,Ordering::Relaxed);state.session.event(None,reason);None}}
    }
}

#[cfg(feature = "ros2")]
pub use integration::{CaptureAuditPlugin, ticket_for_copy};

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(capacity:usize)->(AuditSession,crossbeam_channel::Receiver<WriteJob>) {
        let (tx,rx)=bounded(capacity);
        (AuditSession(Arc::new(SessionInner {id:"offline-fixture".into(),tx:Mutex::new(Some(tx)),
            ledger:Mutex::new(Ledger::default()),counters:Arc::new(Counters::default()),
            closing:std::sync::atomic::AtomicBool::new(false),frame_limit:0,
            trigger_file:None,request_quota:AtomicU64::new(u64::MAX),
            worker:Mutex::new(None),completion:Mutex::new(None)})),rx)
    }
    fn snapshot(extraction:u64)->Arc<AuditFrameSnapshot> {
        Arc::new(AuditFrameSnapshot::new(extraction,42,2,1,
            json!({"fixture_GT":extraction,"evidence_kind":"synthetic_software_fixture","missing_reasons":{}})).unwrap())
    }
    fn submit(ticket:&mut FrameTicket,value:u8)->Result<(),&'static str> {
        ticket.post_noise_rgb(2,1,&[value,2,3,4,5,6],0.,7)
    }
    fn frame(rx:&crossbeam_channel::Receiver<WriteJob>)->(Value,Vec<u8>) {
        match rx.recv().unwrap() {WriteJob::Frame {ticket,rgb,..}=>(ticket,rgb),_=>panic!("expected frame")}
    }
    fn temp_dir()->PathBuf {
        let p=std::env::temp_dir().join(format!("rm-capture-unit-{}-{}",std::process::id(),wall_ns().unwrap()));
        fs::create_dir(&p).unwrap();fs::create_dir(p.join("images")).unwrap();fs::create_dir(p.join("metadata")).unwrap();p
    }

    #[test] fn unique_ticket_normal_association() {
        let (session,rx)=fixture(4);let mut t=session.claim(snapshot(1)).unwrap();
        assert_eq!(t.snapshot.extraction_id,1);assert_eq!(t.packet_id,1);
        submit(&mut t,11).unwrap();let (m,b)=frame(&rx);
        assert_eq!(m["fixture_GT"],1);assert_eq!(m["gpu_copy_packet_id"],1);assert_eq!(b[0],11);
    }
    #[test] fn out_of_order_buffer_completion_preserves_gt() {
        let (s,rx)=fixture(4);let mut a=s.claim(snapshot(1)).unwrap();let mut b=s.claim(snapshot(2)).unwrap();
        submit(&mut b,22).unwrap();submit(&mut a,11).unwrap();
        let (m2,p2)=frame(&rx);let(m1,p1)=frame(&rx);
        assert_eq!((m2["fixture_GT"].as_u64(),p2[0]),(Some(2),22));
        assert_eq!((m1["fixture_GT"].as_u64(),p1[0]),(Some(1),11));
    }
    #[test] fn delayed_old_packet_is_not_given_latest_gt() {
        let (s,rx)=fixture(4);let mut old=s.claim(snapshot(1)).unwrap();
        let newer=s.claim(snapshot(20)).unwrap();drop(newer);
        submit(&mut old,91).unwrap();assert!(matches!(rx.recv().unwrap(),WriteJob::Event(_)));
        assert_eq!(frame(&rx).0["fixture_GT"],1);
    }
    #[test] fn duplicate_frame_ticket_claim_rejected() {
        let (s,_rx)=fixture(4);let _t=s.claim(snapshot(1)).unwrap();
        assert_eq!(s.claim(snapshot(1)).err(),Some("duplicate_or_stale_extraction"));
    }
    #[test] fn duplicate_completion_rejected() {
        let (s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();submit(&mut t,1).unwrap();
        assert_eq!(submit(&mut t,2),Err("duplicate_ticket_completion"));
    }
    #[test] fn missing_gt_ledger_entry_rejected() {
        let(s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();s.0.ledger.lock().unwrap().pending.clear();
        assert_eq!(submit(&mut t,1),Err("packet_snapshot_identity_mismatch"));
    }
    #[test] fn missing_image_rejected() {
        let(s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();
        assert_eq!(t.post_noise_rgb(2,1,&[],0.,7),Err("image_byte_count_mismatch"));
    }
    #[test] fn writer_queue_overflow_nonblocking() {
        let(s,_rx)=fixture(1);let mut a=s.claim(snapshot(1)).unwrap();let mut b=s.claim(snapshot(2)).unwrap();
        submit(&mut a,1).unwrap();assert_eq!(submit(&mut b,2),Err("writer_queue_full"));
        assert_eq!(s.0.counters.queue_full.load(Ordering::Relaxed),1);
        assert_eq!(s.0.counters.event_overflow.load(Ordering::Relaxed),1);
    }
    #[test] fn pending_capacity_and_sequence_gaps_retained() {
        let(s,_rx)=fixture(32);let mut pending=Vec::new();
        for i in 1..=MAX_PENDING {pending.push(s.claim(snapshot(i as u64)).unwrap());}
        assert_eq!(s.claim(snapshot(9)).err(),Some("pending_capacity_exceeded"));
        drop(pending.pop());let next=s.claim(snapshot(10)).unwrap();
        assert_eq!(next.capture_sequence,10);assert_eq!(next.packet_id,9);
    }
    #[test] fn disconnected_writer_does_not_mutate_rgb() {
        let(s,rx)=fixture(4);drop(rx);let mut t=s.claim(snapshot(1)).unwrap();let rgb=[9,8,7,6,5,4];
        assert_eq!(t.post_noise_rgb(2,1,&rgb,0.,7),Err("writer_disconnected"));assert_eq!(rgb,[9,8,7,6,5,4]);
    }
    #[test] fn image_format_error_records_rejection() {
        let(s,rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();t.reject_format();
        if let WriteJob::Event(v)=rx.recv().unwrap() {assert_eq!(v["reason"],"image_format_not_rgb8");}else{panic!()}
    }
    #[test] fn dimensions_mismatch_rejected() {
        let(s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();
        assert_eq!(t.post_noise_rgb(1,2,&[0;6],0.,7),Err("image_dimensions_mismatch"));
    }
    #[test] fn malformed_rgb_length_rejected() {
        let(s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();
        assert_eq!(t.post_noise_rgb(2,1,&[0;5],0.,7),Err("image_byte_count_mismatch"));
    }
    #[test] fn nonfinite_marker_rejected() {
        let mut p=[[0.;3];4];p[0][0]=f64::NAN;assert_eq!(canonical_marker(p).err(),Some("nonfinite_marker"));
        p[0][0]=f64::INFINITY;assert!(canonical_marker(p).is_err());
    }
    #[test] fn center_absent_never_falls_back_to_root() {
        assert_eq!(unique_center::<u64>(&[]),Err("CENTER_missing_no_fallback"));
        assert_eq!(unique_center(&[11_u64]),Ok(11));
        assert_eq!(unique_center(&[11_u64,12]),Err("multiple_CENTER_entities"));
    }
    #[test] fn marker_position_missing_rejected() {
        assert_eq!(require_marker(None).err(),Some("MARKER_POSITION_missing"));
    }
    #[test] fn transform_nan_and_inf_rejected_before_json() {
        let mut m=[[0.;4];4];m[0][2]=f64::NAN;assert_eq!(finite_matrix(&m),Err("nonfinite_transform"));
        m[0][2]=f64::INFINITY;assert!(finite_matrix(&m).is_err());
    }
    #[test] fn nonfinite_noise_rejected_before_json() {
        let(s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();
        assert_eq!(t.post_noise_rgb(2,1,&[0;6],f32::NAN,7),Err("nonfinite_noise_parameter"));
    }
    #[test] fn extraction_mismatch_rejected() {
        let(s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();
        s.0.ledger.lock().unwrap().pending.insert(t.packet_id,(2,42));
        assert_eq!(submit(&mut t,1),Err("packet_snapshot_identity_mismatch"));
    }
    #[test] fn camera_entity_mismatch_rejected() {
        let(s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();
        s.0.ledger.lock().unwrap().pending.insert(t.packet_id,(1,99));
        assert_eq!(submit(&mut t,1),Err("packet_snapshot_identity_mismatch"));
    }
    #[test] fn stale_extraction_cannot_claim_next_image() {
        let(s,_rx)=fixture(4);let _t=s.claim(snapshot(3)).unwrap();
        assert_eq!(s.claim(snapshot(2)).err(),Some("duplicate_or_stale_extraction"));
    }
    #[test] fn gpu_drop_never_reassigns_next_gt() {
        let(s,rx)=fixture(4);drop(s.claim(snapshot(1)).unwrap());
        let mut next=s.claim(snapshot(2)).unwrap();submit(&mut next,22).unwrap();
        assert!(matches!(rx.recv().unwrap(),WriteJob::Event(_)));
        let(m,b)=frame(&rx);assert_eq!(m["gpu_copy_packet_id"],2);assert_eq!(b[0],22);
        assert_eq!(s.0.counters.abandoned.load(Ordering::Relaxed),1);
    }
    #[test] fn original_rgb_unchanged_and_copied_exactly() {
        let(s,rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();let rgb=[255,0,7,11,64,200];let before=rgb;
        t.post_noise_rgb(2,1,&rgb,2.,123).unwrap();assert_eq!(rgb,before);
        let(m,saved)=frame(&rx);assert_eq!(saved,rgb);assert_eq!(m["noise"]["seed"],123);
    }
    #[test] fn png_roundtrip_rgb_and_bgr_channel_semantics() {
        let root=temp_dir();let rgb=[255,0,7,11,64,200];
        write_frame(&root,json!({"frame_uid":"test-rgb","image":{}}),&rgb,2,1).unwrap();
        let decoded=image::open(root.join("images/test-rgb.png")).unwrap().to_rgb8().into_raw();
        assert_eq!(decoded,rgb);let bgr:Vec<_>=decoded.chunks_exact(3).flat_map(|p|[p[2],p[1],p[0]]).collect();
        assert_eq!(bgr,[7,0,255,200,64,11]);
    }
    #[test] fn create_new_prevents_overwrite() {
        let root=temp_dir();let m=json!({"frame_uid":"exclusive","image":{}});
        write_frame(&root,m.clone(),&[1;6],2,1).unwrap();
        assert!(write_frame(&root,m,&[2;6],2,1).is_err());
        assert_eq!(image::open(root.join("images/exclusive.png")).unwrap().to_rgb8().into_raw(),[1;6]);
    }
    #[test] fn filesystem_write_failure_no_committed_metadata() {
        let root=temp_dir();let bad=root.join("absent");
        assert!(write_frame(&bad,json!({"frame_uid":"failed","image":{}}),&[1;6],2,1).is_err());
        assert!(!bad.join("metadata/failed.json").exists());
    }
    #[test] fn metadata_write_failure_leaves_uncommitted_orphan() {
        let root=temp_dir();File::create(root.join("metadata/partial.json")).unwrap();
        assert!(write_frame(&root,json!({"frame_uid":"partial","image":{}}),&[1;6],2,1).is_err());
        assert!(root.join("images/partial.png").exists());
        assert_eq!(fs::metadata(root.join("metadata/partial.json")).unwrap().len(),0);
    }
    fn wait_manifest(root:&Path)->Value {
        for _ in 0..200 {
            if let Ok(bytes)=fs::read(root.join("manifest.json")) {
                if let Ok(value)=serde_json::from_slice(&bytes) {return value;}
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("bounded writer did not close in two seconds");
    }
    #[test] fn background_writer_commits_then_finalizes_without_sender_cycle() {
        let parent=temp_dir();let session=AuditSession::start(&parent).unwrap();
        let root=parent.join(format!("part2c2_dataset_{}",session.0.id));
        let mut ticket=session.claim(snapshot(1)).unwrap();submit(&mut ticket,27).unwrap();
        let uid=ticket.frame_uid.clone();drop(ticket);drop(session);
        let manifest=wait_manifest(&root);assert_eq!(manifest["counters"]["saved"],1);
        assert_eq!(manifest["frames"][0]["frame_uid"],uid);
        let metadata:Value=serde_json::from_slice(&fs::read(root.join(format!("metadata/{uid}.json"))).unwrap()).unwrap();
        assert_eq!(metadata["evidence_kind"],"synthetic_software_fixture");
        assert_eq!(metadata["runtime_verified_same_packet"],false);
    }
    #[test] fn background_write_failure_counted_no_overwrite() {
        let parent=temp_dir();let session=AuditSession::start(&parent).unwrap();
        let root=parent.join(format!("part2c2_dataset_{}",session.0.id));
        let mut ticket=session.claim(snapshot(1)).unwrap();let uid=ticket.frame_uid.clone();
        let protected=root.join(format!("images/{uid}.png"));
        let mut file=create_new(&protected).unwrap();file.write_all(b"protected fixture").unwrap();drop(file);
        submit(&mut ticket,27).unwrap();drop(ticket);drop(session);
        let manifest=wait_manifest(&root);assert_eq!(manifest["counters"]["write_failed"],1);
        assert_eq!(manifest["frames"].as_array().unwrap().len(),0);
        assert_eq!(fs::read(protected).unwrap(),b"protected fixture");
    }
    #[test] fn empty_camera_dimensions_rejected() {
        assert_eq!(AuditFrameSnapshot::new(1,42,0,1,json!({})).err(),Some("empty_image"));
    }
    #[test] fn structural_status_never_claims_runtime_validation() {
        let(s,rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();submit(&mut t,1).unwrap();
        let(m,_)=frame(&rx);assert_eq!(m["sync_status"],"same_extraction_structural");
        assert_eq!(m["runtime_verified_same_packet"],false);
    }
    #[test] fn canonical_plane_preserves_identity_and_residual() {
        let p=[[-0.067,-0.027,0.000002],[-0.067,0.027,-0.000002],
            [0.067,-0.027,-0.000002],[0.067,0.027,0.000002]];
        let(m,o)=canonical_marker(p).unwrap();
        for i in 0..4 {for a in 0..3 {
            let reconstructed=m[a][3]+(0..3).map(|j|m[a][j]*o[i][j]).sum::<f64>();
            assert!((reconstructed-p[i][a]).abs()<1e-12);
        }}
        assert!(o[0][0]<0. && o[2][0]>0. && o[1][1]>0.);
        assert!((o[0][2].abs()-0.000002).abs()<1e-12);
    }
    #[test] fn collinear_and_duplicate_marker_rejected() {
        assert!(canonical_marker([[0.;3];4]).is_err());
        assert!(canonical_marker([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]]).is_err());
    }

    // Offline GLB fixture inherited from Part 2B candidate_geometry.json; never a runtime constant.
    #[test] fn small_actual_asset_canonical_frame_matches_previous_svd() {
        let points:[[f64;3];4]=[[-0.06687040714333264,0.0027063630554037196,0.02689541002439255],[-0.06690013321431379,0.0027021701052392776,-0.026989858964663577],[0.06689709337721582,0.002702547530250411,0.026865877032461193],[0.06687346188159181,0.002705374196496099,-0.02702348123378362]];
        let expected:[[f64;3];4]=[[-0.06687636109369596,-0.026943659104384735,1.7550270898382997e-06],[-0.06689419042634433,0.026941615183105342,-1.754947109460667e-06],[0.06689114269534553,-0.02694365910438474,-1.7548939662086279e-06],[0.06687940882469473,0.026945703025664135,1.7548139858309952e-06]];
        let (_,actual)=canonical_marker(points).unwrap();
        for i in 0..4 {for j in 0..3 {assert!((actual[i][j]-expected[i][j]).abs()<1e-11);}}
    }

    // Offline GLB fixture inherited from Part 2B candidate_geometry.json; never a runtime constant.
    #[test] fn large_actual_asset_canonical_frame_matches_previous_svd() {
        let points:[[f64;3];4]=[[-0.11437147062931838,0.002707549115825769,0.02697342874798636],[-0.11440118712242905,0.0027033543786574854,-0.02691184543483871],[0.11439598484714783,0.002702487488561829,0.026945497530029172],[0.11437234898176737,0.002705313226024515,-0.026943871730981783]];
        let expected:[[f64;3];4]=[[-0.11437368019260262,-0.026943662290816104,1.755231977015494e-06],[-0.11439681755808195,0.02694161516753337,-1.7551853108646472e-06],[0.11439377701804948,-0.026943662290816104,-1.7550986267601573e-06],[0.1143767207326351,0.026945709414098844,1.7550519606075758e-06]];
        let (_,actual)=canonical_marker(points).unwrap();
        for i in 0..4 {for j in 0..3 {assert!((actual[i][j]-expected[i][j]).abs()<1e-11);}}
    }

    #[test] fn sha256_standard_vectors() {
        assert_eq!(sha256(b""),"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256(b"abc"),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        assert_eq!(sha256(&vec![b'a';1_000_000]),"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }
    #[test] fn source_identity_mismatch_fails_closed() {
        assert_eq!(validate_copy_source(&"GPU_target_A",&"GPU_target_B"),Err("registered_ROS_copy_source_mismatch"));
        assert_eq!(validate_copy_source(&"GPU_target_A",&"GPU_target_A"),Ok(()));
    }
    #[test] fn forced_dual_backend_audit_rejected_only_when_applicable() {
        assert!(audit_backend_supported(false,false));assert!(audit_backend_supported(false,true));
        assert!(audit_backend_supported(true,false));assert!(!audit_backend_supported(true,true));
    }
    fn session_root(parent:&Path,session:&AuditSession)->PathBuf {
        parent.join(format!("part2c2_dataset_{}",session.0.id))
    }
    #[test] fn explicit_shutdown_drains_accepted_jobs_with_live_session_clones() {
        let parent=temp_dir();let s=AuditSession::start_with_limit(&parent,6).unwrap();let clone=s.clone();
        let mut a=s.claim(snapshot(1)).unwrap();let mut b=s.claim(snapshot(2)).unwrap();
        submit(&mut b,22).unwrap();submit(&mut a,11).unwrap();
        let m=s.shutdown().unwrap();assert_eq!(m["complete"],true);assert_eq!(m["frames"].as_array().unwrap().len(),2);
        assert_eq!(m["frames"][0]["frame_uid"],b.frame_uid);assert_eq!(m["frames"][1]["frame_uid"],a.frame_uid);
        assert!(s.0.worker.lock().unwrap().is_none());assert_eq!(clone.shutdown().unwrap(),m);
        let root=session_root(&parent,&s);
        for record in m["frames"].as_array().unwrap() {verify_frame_pair(&root,record).unwrap();}
    }
    #[test] fn shutdown_abandons_pending_without_sender_cycle_or_double_count() {
        let parent=temp_dir();let s=AuditSession::start_with_limit(&parent,6).unwrap();
        let mut pending=s.claim(snapshot(1)).unwrap();let m=s.shutdown().unwrap();
        assert_eq!(m["complete"],false);assert_eq!(m["counters"]["abandoned_before_post_noise"],1);
        assert_eq!(submit(&mut pending,7),Err("capture_session_closed"));drop(pending);
        assert_eq!(s.0.counters.abandoned.load(Ordering::Relaxed),1);assert!(s.claim(snapshot(2)).is_err());
    }
    #[test] fn dropped_live_ticket_is_accounted_before_final_manifest() {
        let parent=temp_dir();let s=AuditSession::start_with_limit(&parent,6).unwrap();
        drop(s.claim(snapshot(1)).unwrap());let m=s.shutdown().unwrap();
        assert_eq!(m["complete"],false);assert_eq!(m["counters"]["allocated"],1);
        assert_eq!(m["counters"]["abandoned_before_post_noise"],1);
    }
    #[test] fn final_manifest_rejects_orphan_png_even_if_other_pair_valid() {
        let root=temp_dir();let r=write_frame(&root,json!({"frame_uid":"good","image":{}}),&[4;6],2,1).unwrap();
        fs::write(root.join("images/orphan.png"),b"partial PNG").unwrap();
        let c=Counters::default();for v in [&c.allocated,&c.submitted,&c.saved] {v.store(1,Ordering::Relaxed);}
        let m=finalize_manifest(&root,"fixture",6,&c,vec![r]).unwrap();assert_eq!(m["complete"],false);
    }
    #[test] fn final_manifest_rechecks_png_tampering() {
        let root=temp_dir();let r=write_frame(&root,json!({"frame_uid":"tamper","image":{}}),&[4;6],2,1).unwrap();
        fs::write(root.join("images/tamper.png"),b"modified").unwrap();
        let c=Counters::default();for v in [&c.allocated,&c.submitted,&c.saved] {v.store(1,Ordering::Relaxed);}
        let m=finalize_manifest(&root,"fixture",6,&c,vec![r]).unwrap();assert_eq!(m["complete"],false);
        assert_eq!(m["frames"][0]["complete"],false);assert_eq!(m["pair_validation_failures"].as_array().unwrap().len(),1);
    }
    #[test] fn metadata_uid_and_hash_tampering_rejected() {
        let root=temp_dir();let r=write_frame(&root,json!({"frame_uid":"original","image":{}}),&[4;6],2,1).unwrap();
        let mp=root.join("metadata/original.json");let mut meta:Value=serde_json::from_slice(&fs::read(&mp).unwrap()).unwrap();
        meta["frame_uid"]=json!("different");fs::write(&mp,serde_json::to_vec(&meta).unwrap()).unwrap();
        assert!(verify_frame_pair(&root,&r).is_err());
        let mut rehashed=r;rehashed["metadata_sha256"]=json!(sha256(&fs::read(&mp).unwrap()));
        assert!(verify_frame_pair(&root,&rehashed).unwrap_err().to_string().contains("UID"));
    }
    #[test] fn manifest_write_failure_never_returns_completion() {
        let parent=temp_dir();let s=AuditSession::start_with_limit(&parent,6).unwrap();let root=session_root(&parent,&s);
        fs::write(root.join("manifest.json"),b"preexisting protected fixture").unwrap();
        let mut t=s.claim(snapshot(1)).unwrap();submit(&mut t,3).unwrap();
        assert!(s.shutdown().is_err());assert_eq!(fs::read(root.join("manifest.json")).unwrap(),b"preexisting protected fixture");
    }
    #[test] fn writer_panic_join_is_reported_without_success_manifest() {
        let parent=temp_dir();let s=AuditSession::start_with_limit(&parent,6).unwrap();let root=session_root(&parent,&s);
        s.0.tx.lock().unwrap().as_ref().unwrap().send(WriteJob::PanicForTest).unwrap();
        assert!(s.shutdown().unwrap_err().to_string().contains("panicked"));assert!(!root.join("manifest.json").exists());
    }
    #[test] fn empty_manifest_never_complete() {
        let parent=temp_dir();let s=AuditSession::start_with_limit(&parent,6).unwrap();
        let m=s.shutdown().unwrap();assert_eq!(m["complete"],false);assert_eq!(m["frames"],json!([]));
    }
    #[test] fn six_packet_limit_does_not_renumber_or_require_gui() {
        let (mut s,rx)=fixture(32);Arc::get_mut(&mut s.0).unwrap().frame_limit=6;
        for i in 1..=6 {let mut t=s.claim(snapshot(i)).unwrap();submit(&mut t,i as u8).unwrap();}
        assert_eq!(s.claim(snapshot(7)).err(),Some("capture_closed_or_limit_reached"));
        for i in 1..=6 {assert_eq!(frame(&rx).0["capture_sequence"],i);}
        assert_eq!(s.0.counters.allocated.load(Ordering::Relaxed),6);
    }
    #[test] fn callback_sender_busy_does_not_wait_or_mutate_rgb() {
        let(s,_rx)=fixture(4);let mut t=s.claim(snapshot(1)).unwrap();let rgb=[1,2,3,4,5,6];
        let guard=s.0.tx.lock().unwrap();
        assert_eq!(t.post_noise_rgb(2,1,&rgb,0.,1),Err("capture_sender_busy"));assert_eq!(rgb,[1,2,3,4,5,6]);drop(guard);
    }
    #[test] fn invalid_uid_and_rgb_cannot_create_pair() {
        let root=temp_dir();assert!(write_frame(&root,json!({"frame_uid":"../bad","image":{}}),&[1;6],2,1).is_err());
        assert!(write_frame(&root,json!({"frame_uid":"badlen","image":{}}),&[1;5],2,1).is_err());
    }
    #[test] fn png_json_failure_forces_incomplete_manifest() {
        let parent=temp_dir();let s=AuditSession::start_with_limit(&parent,6).unwrap();let root=session_root(&parent,&s);
        let mut t=s.claim(snapshot(1)).unwrap();fs::write(root.join(format!("metadata/{}.json",t.frame_uid)),b"protected").unwrap();
        submit(&mut t,3).unwrap();let m=s.shutdown().unwrap();assert_eq!(m["complete"],false);
        assert_eq!(m["counters"]["write_failed"],1);assert_eq!(m["frames"],json!([]));
        assert!(root.join(format!("images/{}.png",t.frame_uid)).exists());
    }

    #[test] fn exact_ros_header_stamp_survives_owned_packet() {
        let(s,rx)=fixture(4);
        let stamp=json!({"sec":1794710000,"nanosec":123456789,"clock":"ROS SystemTime", "semantics":"fixture exact ROS header"});
        let snap=Arc::new(AuditFrameSnapshot::new(1,42,2,1,json!({"evidence_kind":"synthetic_software_fixture","ros_image_header_stamp":stamp,"missing_reasons":{}})).unwrap());
        let mut ticket=s.claim(snap).unwrap();submit(&mut ticket,3).unwrap();
        assert_eq!(frame(&rx).0["ros_image_header_stamp"],stamp);
    }

    #[test] fn manual_request_is_disarmed_until_valid_and_never_reuses_quota() {
        let (mut s,rx)=fixture(8);let root=temp_dir();let path=root.join("request.txt");
        Arc::get_mut(&mut s.0).unwrap().trigger_file=Some(path.clone());
        s.refresh_request_quota();assert!(!s.accepting());
        fs::write(&path,b"1\n").unwrap();s.refresh_request_quota();
        let mut t=s.claim(snapshot(1)).unwrap();submit(&mut t,3).unwrap();
        assert!(!s.accepting());assert_eq!(frame(&rx).0["pilot_control"]["requested_capture_quota"],1);
        fs::write(&path,b"2\n").unwrap();s.refresh_request_quota();assert!(s.accepting());
        fs::write(&path,b"not a number").unwrap();s.refresh_request_quota();assert!(!s.accepting());
    }
    #[test] fn oversized_request_file_fails_closed() {
        let(mut s,_rx)=fixture(4);let root=temp_dir();let path=root.join("request.txt");
        Arc::get_mut(&mut s.0).unwrap().trigger_file=Some(path.clone());
        fs::write(path,vec![b'1';65]).unwrap();s.refresh_request_quota();assert!(!s.accepting());
    }

}
