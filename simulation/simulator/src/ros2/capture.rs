//! ROS2 图像捕获实现

use crate::capture::{
    CameraFov, CaptureBundle, ImageHandle, compute_camera_intrinsics,
    driver::{
        CaptureConfig, CapturedFrame, CapturedFrameKind, GpuCaptureHandler, SnapshotAsync,
        SnapshotSync,
    },
    setup_capture_camera, setup_preview_window, sync_capture_camera,
};
use crate::dataset::prelude::DatasetSnapshotCreator;
use crate::ros2::image::compress_image;
use crate::ros2::topic::{CameraInfoTopic, ImageCompressedTopic, ImageRawTopic, TopicPublisher};
use crate::systems::GameplaySystems;
use bevy::ecs::world::DeferredWorld;
use bevy::prelude::*;
use bevy::render::{RenderApp, RenderSystems};
use r2r::Clock;
use r2r::sensor_msgs::msg::{CameraInfo, RegionOfInterest};
use r2r::std_msgs::msg::Header;
use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock};

struct RosSnapshotSync {
    stamp: RefCell<r2r::builtin_interfaces::msg::Time>,
    audit_ticket: Option<crate::capture_audit::FrameTicket>,
}

impl SnapshotSync for RosSnapshotSync {
    fn captured(
        self: Box<Self>,
        world: &mut DeferredWorld,
        _config: &CaptureConfig,
    ) -> Box<dyn SnapshotAsync> {
        // The snapshot runs on the capture task pool, where only resources carried over
        // by the capture plugin exist; the sensor-noise setting travels in the context.
        let ctx = world.resource::<RosCaptureContextShared>().0.clone();
        Box::new(RosSnapshot {
            stamp: self.stamp,
            audit_ticket: self.audit_ticket,
            ctx,
        })
    }
}

struct RosSnapshot {
    stamp: RefCell<r2r::builtin_interfaces::msg::Time>,
    audit_ticket: Option<crate::capture_audit::FrameTicket>,
    ctx: Arc<RosCaptureContext>,
}

impl SnapshotAsync for RosSnapshot {
    fn captured(&mut self, frame: CapturedFrame<'_>) {
        if frame.kind != CapturedFrameKind::Rgb8 {
            if let Some(ticket) = &mut self.audit_ticket {
                ticket.reject_format();
            }
            return;
        }

        // A real sensor never delivers a frame this clean; the noise is applied to what
        // subscribers see, so a stack validated here is not depending on a constant
        // background. Zero by default, which leaves `frame.data` untouched.
        let mut noisy;
        let data = if self.ctx.noise_sigma > 0.0 {
            noisy = frame.data.to_vec();
            let stamp = self.stamp.borrow();
            crate::vision_fidelity::add_sensor_noise(
                &mut noisy,
                self.ctx.noise_sigma,
                stamp.sec as u32 ^ (stamp.nanosec >> 8),
            );
            &noisy[..]
        } else {
            frame.data
        };

        // This is the very same post-noise RGB slice passed to raw_image below.
        // Export is nonblocking and failure never changes the original publication.
        if let Some(ticket) = &mut self.audit_ticket {
            let stamp = self.stamp.borrow();
            let seed = stamp.sec as u32 ^ (stamp.nanosec >> 8);
            let _ = ticket.post_noise_rgb(
                frame.width, frame.height, data, self.ctx.noise_sigma, seed,
            );
        }

        let optical_frame_hdr = Header {
            stamp: self.stamp.take(),
            frame_id: "camera_optical_frame".to_string(),
        };
        self.ctx.camera_info.publish(ros_camera_info(
            optical_frame_hdr.clone(),
            frame.width,
            frame.height,
            self.ctx.fov_y,
        ));
        if self.ctx.publish_compressed {
            self.ctx.image_compressed.publish(compress_image(
                optical_frame_hdr,
                frame.width,
                frame.height,
                data,
            ));
        } else {
            self.ctx.image_raw.publish(raw_image(
                optical_frame_hdr,
                frame.width,
                frame.height,
                data,
            ));
        }
    }
}

#[derive(Default)]
struct RosSnapshotCreator {
    // Set once to CaptureBundle's actual color target, independent of ImageHandle.
    audit_source: Arc<OnceLock<Handle<Image>>>,
}

impl GpuCaptureHandler for RosSnapshotCreator {
    fn captured(&self, world: &World) -> Option<Box<dyn SnapshotSync>> {
        let clock = world.resource::<RosCaptureContext>();
        let stamp = Clock::to_builtin_time(&clock.clock.lock().unwrap().get_now().unwrap());
        Some(Box::new(RosSnapshotSync {
            stamp: RefCell::new(stamp.clone()),
            // The existing driver puts this object in the same tuple as the
            // actual copy destination buffer; no FIFO/time-based metadata lookup.
            audit_ticket: crate::capture_audit::ticket_for_copy(world, self.audit_source.get(), stamp.sec, stamp.nanosec),
        }))
    }
}

#[derive(Resource, Clone, Deref, DerefMut)]
pub struct RosCaptureContextShared(Arc<RosCaptureContext>);

#[derive(Resource, Clone)]
pub struct RosCaptureContext {
    pub clock: Arc<Mutex<Clock>>,
    pub fov_y: f32,
    pub publish_compressed: bool,
    /// Gaussian sensor noise in gray levels added to published frames (`[vision]`).
    pub noise_sigma: f32,
    pub camera_info: TopicPublisher<CameraInfoTopic>,
    pub image_raw: TopicPublisher<ImageRawTopic>,
    pub image_compressed: TopicPublisher<ImageCompressedTopic>,
}

#[derive(Resource)]
pub(crate) struct RosAuditCopySource(Handle<Image>);
impl RosAuditCopySource {
    pub(crate) fn handle(&self) -> &Handle<Image> { &self.0 }
}

pub struct RosCapturePlugin {
    pub config: CaptureConfig,
    pub context: RosCaptureContext,
}

impl Plugin for RosCapturePlugin {
    fn build(&self, app: &mut App) {
        let audit_source = Arc::new(OnceLock::new());
        let capture = CaptureBundle::color_and_depth(
            app,
            self.config.clone(),
            vec![
                Box::new(RosSnapshotCreator { audit_source: audit_source.clone() }),
                Box::new(DatasetSnapshotCreator::default()),
            ],
            vec![Box::new(DatasetSnapshotCreator::depth())],
        );
        let render_target_handle = capture.color_target().unwrap().clone();
        audit_source.set(render_target_handle.clone()).expect("ROS copy source initialized once");

        app.add_plugins(capture)
            .insert_resource(RosAuditCopySource(render_target_handle.clone()))
            .insert_resource(ImageHandle(render_target_handle))
            .insert_resource(CameraFov(self.context.fov_y))
            .insert_resource(self.context.clone())
            .add_systems(Startup, setup_capture_camera)
            .add_systems(Startup, setup_preview_window)
            .add_systems(
                Update,
                sync_capture_camera
                    .after(GameplaySystems::Camera)
                    .before(RenderSystems::Render),
            );
        app.sub_app_mut(RenderApp)
            .insert_resource(RosCaptureContextShared(Arc::new(self.context.clone())))
            .insert_resource(self.context.clone());
    }
}

fn raw_image(hdr: Header, width: u32, height: u32, data: &[u8]) -> r2r::sensor_msgs::msg::Image {
    r2r::sensor_msgs::msg::Image {
        header: hdr,
        height,
        width,
        encoding: "rgb8".to_string(),
        is_bigendian: 0,
        step: width * 3,
        data: Vec::from(data),
    }
}

fn ros_camera_info(hdr: Header, width: u32, height: u32, fov_y: f32) -> CameraInfo {
    let intrinsics = compute_camera_intrinsics(width, height, fov_y);

    CameraInfo {
        header: hdr,
        height,
        width,
        distortion_model: "plumb_bob".to_string(),
        d: vec![0.000, 0.000, 0.000, 0.000, 0.000],
        k: vec![
            intrinsics.fx,
            0.0,
            intrinsics.cx,
            0.0,
            intrinsics.fy,
            intrinsics.cy,
            0.0,
            0.0,
            1.0,
        ],
        p: vec![
            intrinsics.fx,
            0.0,
            intrinsics.cx,
            0.0,
            0.0,
            intrinsics.fy,
            intrinsics.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        r: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        binning_x: 0,
        binning_y: 0,
        roi: RegionOfInterest {
            x_offset: 0,
            y_offset: 0,
            height,
            width,
            do_rectify: true,
        },
    }
}
