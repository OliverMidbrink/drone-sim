extends Camera3D

@export var move_speed: float = 15.0
@export var sprint_multiplier: float = 4.0
@export var mouse_sensitivity: float = 0.002

var mouse_captured := true


func _ready() -> void:
	current = true
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey:
		if event.pressed and event.keycode == KEY_ESCAPE:
			mouse_captured = not mouse_captured

			Input.mouse_mode = (
				Input.MOUSE_MODE_CAPTURED
				if mouse_captured
				else Input.MOUSE_MODE_VISIBLE
			)

	if event is InputEventMouseMotion and mouse_captured:
		rotate_y(-event.relative.x * mouse_sensitivity)
		rotate_object_local(
			Vector3.RIGHT,
			-event.relative.y * mouse_sensitivity
		)

		rotation.x = clamp(
			rotation.x,
			deg_to_rad(-89),
			deg_to_rad(89)
		)


func _process(delta: float) -> void:
	var direction := Vector3.ZERO

	if Input.is_key_pressed(KEY_W):
		direction -= global_transform.basis.z

	if Input.is_key_pressed(KEY_S):
		direction += global_transform.basis.z

	if Input.is_key_pressed(KEY_A):
		direction -= global_transform.basis.x

	if Input.is_key_pressed(KEY_D):
		direction += global_transform.basis.x

	if Input.is_key_pressed(KEY_SPACE):
		direction += Vector3.UP

	if Input.is_key_pressed(KEY_CTRL):
		direction -= Vector3.UP

	direction = direction.normalized()

	var speed = move_speed

	if Input.is_key_pressed(KEY_SHIFT):
		speed *= sprint_multiplier

	global_position += direction * speed * delta
