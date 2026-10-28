extends Node3D

const DRONE_MODEL = preload("res://drone-stuff/5inchQuadcopter.glb")

var drone: RigidBody3D
var tcp := StreamPeerTCP.new()
var streaming := false

func _ready():
	spawn_drone()
	start_streaming()

func spawn_drone():
	drone = RigidBody3D.new()
	drone.name = "Drone"
	drone.position = Vector3(0, 10, 0)
	
	# Temporary hover mode
	drone.freeze = true
	add_child(drone)

	var model = DRONE_MODEL.instantiate()
	model.scale = Vector3.ONE * 0.001
	drone.add_child(model)

	# Add collision
	var collision = CollisionShape3D.new()
	var shape = BoxShape3D.new()
	shape.size = Vector3(0.4, 0.15, 0.4)
	collision.shape = shape
	drone.add_child(collision)

	spawn_cameras()


func spawn_cameras():
	create_camera("LeftCamera", -90)
	create_camera("RightCamera", 90)


func create_camera(camera_name: String, angle: float):
	var viewport = SubViewport.new()
	viewport.name = camera_name + "Viewport"
	viewport.size = Vector2i(512, 512)
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS

	add_child(viewport)
	viewport.world_3d = get_viewport().world_3d

	var camera = Camera3D.new()
	camera.name = camera_name
	camera.fov = 120
	camera.rotation_degrees.y = angle

	viewport.add_child(camera)
	

func _process(_delta):
	if drone == null:
		return

	for camera_name in ["LeftCamera", "RightCamera"]:
		var viewport = get_node(camera_name + "Viewport")
		var camera = viewport.get_node(camera_name)

		camera.global_position = drone.global_position
		camera.global_rotation = drone.global_rotation

		camera.rotate_y(
			deg_to_rad(-90 if camera_name == "LeftCamera" else 90)
		)
		
		
func start_streaming() -> void:
	var err = tcp.connect_to_host("127.0.0.1", 5555)

	if err != OK:
		push_error("Could not connect to Python")
		return

	streaming = true
	stream_loop()


func stream_loop() -> void:
	while streaming and is_inside_tree():
		tcp.poll()

		if tcp.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			# Wait until cameras have rendered
			await RenderingServer.frame_post_draw

			send_camera_frame("LeftCameraViewport", 0)
			send_camera_frame("RightCameraViewport", 1)

		elif tcp.get_status() == StreamPeerTCP.STATUS_ERROR:
			push_error("Python connection lost")
			streaming = false
			break

		# Approximately 5 FPS
		await get_tree().create_timer(0.2).timeout


func send_camera_frame(viewport_name: String, camera_id: int) -> void:
	var viewport: SubViewport = get_node(viewport_name)

	var image = viewport.get_texture().get_image()

	if image.is_empty():
		return

	var jpg = image.save_jpg_to_buffer(0.8)

	# Camera ID (1 byte) + JPEG length (4 bytes, little-endian)
	var header = PackedByteArray()
	header.resize(5)
	header[0] = camera_id
	header.encode_u32(1, jpg.size())

	if tcp.put_data(header) != OK:
		streaming = false
		return

	if tcp.put_data(jpg) != OK:
		streaming = false


func _exit_tree() -> void:
	streaming = false
	tcp.disconnect_from_host()
