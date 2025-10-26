import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3
from pyglet.window import key, mouse
from pyglet.graphics.shader import Shader, ShaderProgram
import numpy as np
import os
# Agregar mas si es necesario

# Rutas
root = os.path.dirname(__file__)

# Librerías propias
from librerias.scene_graph import *
from librerias.helpers import mesh_from_file
from librerias.drawables import Model
from librerias import shapes

#Música
# Cargar y reproducir música de fondo, al igual que en tareas anteriores
musica_fondo = pyglet.media.load(root + "/assets/cantinaband.m4a")
player_musica = pyglet.media.Player()
player_musica.queue(musica_fondo)
player_musica.loop = True
player_musica.play()

# --------------------------
# Ventana principal
# --------------------------
class Controller(pyglet.window.Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time = 0.0
        self.gameState = 0
        self.exclusive = False #Modo de mouse exclusivo
        self.set_exclusive_mouse(self.exclusive) #Configura el mouse en modo exclusivo
        #Modo exclusivo significa que el cursor se oculta, y no se puede mover fuera de la ventana de la aplicacion
        # Agregar más atributos si se necesita (como cámara actual, inputs, etc.)

WIDTH = 1000
HEIGHT = 1000
window = Controller(WIDTH, HEIGHT, "Katamari Damacy de Aliexpress")

# --------------------------
# Shaders
# --------------------------
if __name__ == "__main__":
    # Shader con textura e ilumuinacion
    vert_source = """#version 330
    in vec3 position;
    in vec3 normal;
    in vec2 texCoord; // Coords de textura
    
    out vec2 fragTexCoord;
    out vec3 fragNormal;
    out vec3 fragPos;

    uniform mat4 u_model = mat4(1.0); // Matriz de modelo
    uniform mat4 u_view = mat4(1.0); // Matriz de vista
    uniform mat4 u_projection = mat4(1.0); // Matriz de proyeccion

    void main() {
        fragTexCoord = texCoord; // Asigna las coords de textura
        fragNormal = mat3(transpose(inverse(u_model))) * normal; // Calcula la normal transformada
        fragPos = vec3(u_model * vec4(position, 1.0)); // Posicion transformada
        gl_Position = u_projection * u_view * vec4(fragPos, 1.0); // Posicion final del vertice
    }
    """

    frag_source = """#version 330
    in vec2 fragTexCoord;
    in vec3 fragNormal;
    in vec3 fragPos;

    uniform sampler2D u_texture; // Textura
    uniform vec3 lightDir = normalize(vec3(-1.0, -1.0, -0.5)); // Direccion de la luz
    uniform vec3 lightColor = vec3(1.0, 1.0, 1.0); // Color de la luz (blanca)
    uniform vec3 ambientColor = vec3(0.3, 0.3, 0.3); // Color ambiental
    uniform vec3 u_emission = vec3(0.0);  // Emisión añadida

    out vec4 outColor; // Color de salida

    void main() {
        vec3 norm = normalize(fragNormal);
        float diff = max(dot(norm, -lightDir), 0.0); // Calculo de luz difusa
        vec3 diffuse = diff * lightColor; // Color difuso
        vec3 ambient = ambientColor; // Color ambiental
        vec3 texColor = texture(u_texture, fragTexCoord).rgb; // Color de la textura
        vec3 finalColor = (ambient + diffuse) * texColor + u_emission;  // Emisión aplicada
        outColor = vec4(finalColor, 1.0);
    }
    """
    
    # Shader con color sólido, los cambios de iluminacion son analogos a lo anterior
    color_vert = """#version 330
    in vec3 position;
    in vec3 normal;

    out vec3 fragNormal;
    out vec3 fragPos;

    uniform mat4 u_model = mat4(1.0);
    uniform mat4 u_view = mat4(1.0);
    uniform mat4 u_projection = mat4(1.0);

    void main() {
        fragNormal = mat3(transpose(inverse(u_model))) * normal;
        fragPos = vec3(u_model * vec4(position, 1.0));
        gl_Position = u_projection * u_view * vec4(fragPos, 1.0);
    }
    """
    
    color_frag = """#version 330
    in vec3 fragNormal;
    in vec3 fragPos;

    uniform vec4 u_color;
    uniform vec3 lightDir = normalize(vec3(-1.0, -1.0, -0.5));
    uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
    uniform vec3 ambientColor = vec3(0.3, 0.3, 0.3);

    out vec4 outColor;

    void main() {
        vec3 norm = normalize(fragNormal);
        float diff = max(dot(norm, -lightDir), 0.0);
        vec3 diffuse = diff * lightColor * u_color.rgb;
        vec3 ambient = ambientColor * u_color.rgb;
        outColor = vec4(ambient + diffuse, u_color.a);
    }
    """
    # Pipelines
    #Pipeline de las figuras con texturas, incluye la iluminacion
    pipeline = ShaderProgram(Shader(vert_source, "vertex"), Shader(frag_source, "fragment"))
    color_pipeline = ShaderProgram(Shader(color_vert, "vertex"), Shader(color_frag, "fragment"))

    pipeline["u_emission"] = (0.0, 0.0, 0.0)  # Inicializar emisión DE LUZ

    # --------------------------
    # Cargar modelos
    # --------------------------
    #Planicie (lo deje tal cual estaba)
    grass = Texture(root + "/assets/grass.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)
    face_uv = [0, 0, 1, 0, 1, 1, 0, 1]
    texcoords = face_uv * 6
    cube = Model(shapes.Cube["position"], texcoords, index_data=shapes.Cube["indices"], normal_data=shapes.Cube["normal"])

    #Esfera
    sphere = mesh_from_file(root + "/assets/ball/core_01.obj")
    #Roca (obj1)
    rock = mesh_from_file(root + "/assets/Rock_1_I_Color1.obj")
    #Arbol (obj2)
    tree = mesh_from_file(root + "/assets/Tree_1_A_Color1.obj")
    #Hongo rojo (obj3)
    mushroom = mesh_from_file(root + "/assets/mushroom/Mushroom.obj")
    #LEGO Batman (obj4)
    batman = mesh_from_file(root + "/assets/batman/Batman.obj")
    #Sif - Dark Souls (obj5)
    sif = mesh_from_file(root + "/assets/sif/c5210.obj")
    #Mega Hongo (obj6)
    mmushroom = mesh_from_file(root + "/assets/mmushroom/Mushroom.obj")
    #Master Chief (personaje "empujando")
    halo = mesh_from_file(root + "/assets/masterchief/Master Chief Rotado.obj")

    # --------------------------
    # Grafo de escena
    # --------------------------
    world = SceneGraph()
    world.add_node("scene")

    #Planicie
    tile_size = 1.0 
    grid_size = 10  # 10x10
    for i in range(-grid_size // 2, grid_size // 2):
        for j in range(-grid_size // 2, grid_size // 2):
            tile_name = f"grass_{i}_{j}"
            world.add_node(tile_name, mesh=cube, texture=grass, pipeline=pipeline)
            world[tile_name]["position"] = [i * tile_size, -0.01, j * tile_size]
            world[tile_name]["scale"] = [tile_size, 0.01, tile_size]

    #Esfera
    world.add_node("sphere")
    #Seteamos la posicion y la escala a parte de la generacion de los "hijos" del .obj
    world["sphere"]["position"] = [0, 0.3, 0]
    world["sphere"]["scale"] = [0.4, 0.4, 0.4]
    #Como ahora la esfera es un modelo y no una figura generada por nosotros, hay que hacer el mismo procedimiento que la roca
    for part in sphere:
        node_name = f"sphere_{part['id']}"
        world.add_node(node_name, attach_to="sphere", mesh=part["mesh"], pipeline=pipeline, texture=part["texture"])

    #Como vamos a hacer este proceso 6 veces más, mejor usar una función (ya hice el de la esfera, lo voy a dejar así)
    def agregar_modelo(nombre: str, modelo: list, posicion, escala):
        world.add_node(nombre)
        world[nombre]["position"] = posicion
        world[nombre]["scale"] = escala
        for parte in modelo:
            node_id = f"{nombre}_{parte['id']}"
            world.add_node(node_id, attach_to=nombre, mesh=parte["mesh"], pipeline=pipeline, texture=parte["texture"])

    #Llamamos a la función para cargar a los modelos con sus tamaños respectivos
    agregar_modelo("rock", rock, [0.5, 0.4, 0.5], [0.1, 0.1, 0.1])
    agregar_modelo("tree", tree, [2.0, 0.3, 1.5], [0.3, 0.3, 0.3])
    agregar_modelo("mushroom", mushroom, [-2.0, 0.3, 1.5], [0.05, 0.05, 0.05])
    agregar_modelo("batman", batman, [-3.0, 0.3, -1.5], [0.07, 0.07, 0.07])
    agregar_modelo("sif", sif, [0.0, 0.3, -1.0], [0.07, 0.07, 0.07])
    agregar_modelo("mmushroom", mmushroom, [2.0, 0.3, -1.0], [0.03, 0.03, 0.03])
    agregar_modelo("masterchief", halo, [0.0, 0.0, 1.0], [0.015, 0.015, 0.015])

    #Lista para los objetos que puede tomar la pelota
    objetos = ["rock", "tree", "mushroom", "batman", "sif", "mmushroom"]
	
	#Configura la proyeccion de la luz
    pipeline["u_projection"] = Mat4.perspective_projection(WIDTH / HEIGHT, 0.01, 100, 90)
    #Configura la proyeccion para el pipeline de color
    color_pipeline["u_projection"] = pipeline["u_projection"]

    pressed_keys = set() #"Lista" para ver cuáles teclas están presionadas

    # --------------------------
    # Cámara
    # --------------------------
    # Cámara aérea fija
    camera_mode = 1 #Seteamos la cámara a la vista desde arriba

    camera_yaw = 0.0 #Movimiento de la camara horizontalmente
    camera_pitch = 0.0 #Movimiento de la camara verticalmente

    emission_timer = 0.0 #Temporizador para la emision de luz desde la esfera al tocar otros objetos
    emission_duration = 0.5 #Duracion de la emision

    def update_camera():
        sphere_pos = Vec3(*world["sphere"]["position"])
        global camera_yaw, camera_pitch
        if camera_mode == 1: # Cámara desde arriba
            eye = Vec3(sphere_pos.x, 4.0, sphere_pos.z) #Posicion de la camara
            target = Vec3(sphere_pos.x, 0.0, sphere_pos.z) # Centro de la escena
            up = Vec3(0.0, 0.0, -1.0) # "Arriba" apunta hacia -Z para mantener orientación horizontal
        elif camera_mode == 2: #Primera persona
            forward = Vec3(
                np.cos(camera_yaw) * np.cos(camera_pitch),
                0.0,
                np.sin(camera_yaw) * np.cos(camera_pitch)
            ).normalize() #Direccion hacia adelante
            eye = sphere_pos + Vec3(0.0, 0.3, 0.0)
            target = eye + forward #Nos importa lo que esta delante de la bola
            up = Vec3(0.0, 1.0, 0.0)
        elif camera_mode == 3: #Tercera persona
            behind = Vec3(
                -np.cos(camera_yaw) * np.cos(camera_pitch),
                np.sin(camera_pitch),
                -np.sin(camera_yaw) * np.cos(camera_pitch)
            ).normalize() #Direccion hacia atras
            eye = sphere_pos + behind * 3.0 + Vec3(0.0, 1.0, 0.0) # Queremos estar detras de la bola, sin importar como se mueva
            target = sphere_pos #La camara siempre debe mirar hacia la esfera
            up = Vec3(0.0, 1.0, 0.0) #El mismo que primera persona
        view = Mat4.look_at(eye, target, up) #Crea la matriz de vista
        pipeline["u_view"] = view #Asigna la matriz de vista al pipeline
        color_pipeline["u_view"] = view #Asigna la matriz de vista al pipeline de color

    # --------------------------
    # Update y render
    # --------------------------
    @window.event
    def on_key_press(symbol, modifiers):
        global camera_mode
        #Aqui cambiamos los modos de camara con una variable global que revisamos en todo momento
        if symbol == key._1: 
            camera_mode = 1
        elif symbol == key._2:
            camera_mode = 2
        elif symbol == key._3:
            camera_mode = 3
        elif symbol == key.ESCAPE: #Para terminar el programa, lo voy a agregar al README
            window.close()
        pressed_keys.add(symbol) #Para las teclas de movimiento

    @window.event
    def on_key_release(symbol, modifiers):
        #Solo eliminamos las teclas que no estamos presionando
        if symbol in pressed_keys:
            pressed_keys.discard(symbol)

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        #El movimiento del mouse para mover la camara en primera y tercera persona
        global camera_yaw, camera_pitch
        if camera_mode in (2, 3):
            sensitivity = 0.005 #Define que tanto se mueve la camara (la sensibilidad al mover el mouse)
            camera_yaw += dx * sensitivity
            camera_pitch -= dy * sensitivity
            #La camara se mueve distinto segun el modo puesto
            if camera_mode == 2:
  		 	 #Podemos mover la camara libremente
                camera_pitch = np.clip(camera_pitch, -np.pi / 4, np.pi / 4)
            elif camera_mode == 3:
                #No deberiamos poder mover la camara tal que mire hacia arriba, pues la diferencia hace que se ponga debajo del mapa y no podamos ver nada
                #Esto se soluciona en otros juegos acercando la camara para no dejar que el jugador mire fuera del mapa, en vez de simplemente bloquear la camara
                camera_pitch = np.clip(camera_pitch, 0.0, np.pi / 4)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        #Esto es por si te sales de la ventana mientras la estas ejecutando (me paso muchas veces la verdad)
        if button == mouse.RIGHT:
            #Esto es por si necesitas usar el mouse para otra cosa, lo voy a poner en el README
            window.exclusive = not window.exclusive
            window.set_exclusive_mouse(window.exclusive)

    def update(dt):
        #  lógica de juego (colisiones, movimiento, cámara, etc.)
        global emission_timer
        world.update() #Actualiza los nodos del grafo de escena
        sphere_pos = np.array(world["sphere"]["position"], dtype=np.float32) #Posicion de la esfera
        speed = 2.0 #Velocidad con la que se mueve la esfera
        dir_vec = np.zeros(3) #Vector en el que se mueve la esfera (inicializado)
        #Dependiendo de que tecla presionamos, el vector suma en x (A, D) o en z (W, S)
        if key.W in pressed_keys: dir_vec += [0, 0, -1]
        if key.S in pressed_keys: dir_vec += [0, 0, 1]
        if key.A in pressed_keys: dir_vec += [-1, 0, 0]
        if key.D in pressed_keys: dir_vec += [1, 0, 0]

        if camera_mode == 2:
            #Movimiento en primera persona, utilizando el angulo de la camara para definir la direccion
            direction = Vec3(0, 0, 0)
            forward = Vec3(np.cos(camera_yaw), 0.0, np.sin(camera_yaw)) #Vector hacia adelante segun el yaw (angulo)
            right = Vec3(-forward.z, 0, forward.x) #Vector hacia la derecha relativo al forward (para el movimiento horizontal)
            if key.W in pressed_keys: direction += forward
            if key.S in pressed_keys: direction -= forward
            if key.A in pressed_keys: direction -= right
            if key.D in pressed_keys: direction += right
            if direction.length() > 0:
                direction = direction.normalize() #Normaliza la direccion para el mov uniforme
                movement = speed * dt #Desplazamiento en esta direccion
                sphere_pos += np.array([direction.x, 0.0, direction.z]) * movement #Actualiza la posicion de la esfera sumando el mov en los ejes
        else: #Para movimientos en camara 1 y 3
            if np.linalg.norm(dir_vec) > 0:
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                movement = speed * dt
                sphere_pos += dir_vec * movement
                radius = 0.4
                angle = movement / radius
                if "rotation" not in world["sphere"]:
                    #Inicializa la rotacion en caso de no ser un atributo
                    world["sphere"]["rotation"] = [0.0, 0.0, 0.0]
                #Rota la esfera al rededor del eje correspondiente al movimiento
                if abs(dir_vec[2]) > 0.5: #En eje x
                    world["sphere"]["rotation"][0] += angle * np.sign(dir_vec[2])
                elif abs(dir_vec[0]) > 0.5: #En eje z
                    world["sphere"]["rotation"][2] -= angle * np.sign(dir_vec[0])

        world["sphere"]["position"] = sphere_pos.tolist()
        update_camera()
		
		#Distancia limite entre la esfera y un objeto tal que este no lo agarra
        umbral = 0.5
        
        #Veamos que ocurre si el objeto es tomado por la esfera
        for obj in objetos:
            obj_pos = np.array(world[obj]["position"], dtype=np.float32)
            dist = np.linalg.norm(sphere_pos[[0, 2]] - obj_pos[[0, 2]]) #Restamos la distancia de la esfera menos la del objeto para ver lo cerca o lejos que esta
            if dist < umbral:
                #Obtenemos el nodo padre del objeto en el grafo
                padres = list(world.graph.predecessors(obj))
                if len(padres) > 0 and padres[0] != "sphere":
                    #Como el padre no es la esfera, añade a este objeto como hijo de la esfera
                    padre_actual = padres[0]
                    world.graph.remove_edge(padre_actual, obj)
                    world.graph.add_edge("sphere", obj)
                    #Obtiene la posicion relativa para posicionarlo localmente (en el punto donde se tocaron)
                    dir_vec_obj = obj_pos - sphere_pos
                    if np.linalg.norm(dir_vec_obj) > 0:
                        dir_vec_obj /= np.linalg.norm(dir_vec_obj)
                    else:
                        dir_vec_obj = np.array([0,1,0])
                    pos_local = dir_vec_obj * (0.4 + 0.05)
                    world[obj]["position"] = pos_local.tolist() #Actualiza la posicion del objeto c/r a la esfera
                    world[obj]["scale"] = [0.1,0.1,0.1] # Reduce la escala del objeto para que parezca que esta siendo recogido
                    world[obj]["rotation"] = [0.0, 0.0, 0.0] #Resetea la orientacion del objeto
                    emission_timer = emission_duration  # Activa emisión

        if emission_timer > 0:
            #Cuando el timer de emision se activa, se actualiza la intensidad en el tiempo, en este caso se va disminuyendo
            strength = emission_timer / emission_duration
            pipeline["u_emission"] = (1.0 * strength, 0.8 * strength, 0.4 * strength) #Color de emision progresivo
            emission_timer -= dt #Disminuye el tiempo
        else:
            pipeline["u_emission"] = (0.0, 0.0, 0.0) #Apaga la emision
        
         # Master Chief persigue la esfera
        chief_pos = np.array(world["masterchief"]["position"], dtype=np.float32)
        sphere_pos_2d = np.array(world["sphere"]["position"], dtype=np.float32)
        dir_to_sphere = sphere_pos_2d - chief_pos #Distancia entre el y la esfera
        dir_to_sphere[1] = 0  # Solo movimiento horizontal

        dist = np.linalg.norm(dir_to_sphere) #Distancia entre Master Chief y la esfera
        if dist > 0.5:  # Si está suficientemente lejos, se mueve
            dir_norm = dir_to_sphere / dist
            move_speed = 2.0  # Ajusto la velocidad a la misma de la esfera, para que sea un movimiento mas natural
            chief_pos += dir_norm * move_speed * dt #Actualiza la posicion, acercandose a la esfera
            world["masterchief"]["position"] = chief_pos.tolist() #Actualiza la posicion en el grafo
            

            # Rotación en Y para mirar hacia la esfera, esto hace que rote en su propio eje como loco
            angle = np.degrees(np.arctan2(-dir_norm[0], -dir_norm[2]))  # Ojo con z
            world["masterchief"]["rotation"] = [0.0, angle, 0.0]


    @window.event
    def on_draw():
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glClearColor(0.63, 0.6, 0.8, 1.0) # Color fondo
        window.clear()
        world.draw()

    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()
