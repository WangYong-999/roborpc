
from roborpc.cameras.multi_cameras import MultiCameras

if __name__ == '__main__':
    multi_camera = MultiCameras()
    multi_camera.connect_now()
    print(multi_camera.get_device_ids())
    print(multi_camera.get_camera_intrinsics())
    print(multi_camera.get_camera_extrinsics())
    print(multi_camera.read_camera())