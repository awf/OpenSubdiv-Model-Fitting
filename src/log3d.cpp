

#include "log3d.h"

#include <iostream>

int log3d::next_obj = 0;

log3d::log3d(std::string filename, std::string tag) :
  f(filename)
{
  if (!f.good())
    throw std::exception("zoiks");

  next_obj = 0;

  head(tag.length() ? tag : filename);
  canvas();
}

log3d::~log3d()
{
  endcanvas();
  f << "Ended.\n";
  done();
}

void log3d::head(std::string tag) {
  f << R"(<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />

                                                <title>log3d [)" << tag << R"(]</title>
        <!-- Babylon.js -->
        <script src="http://www.babylonjs.com/hand.minified-1.2.js"></script>
        <script src="http://www.babylonjs.com/cannon.js"></script>
        <script src="http://www.babylonjs.com/oimo.js"></script>
        <script src="http://www.babylonjs.com/babylon.js"></script>
        <style>
            html, body {
                overflow: hidden;
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
            }

                canvas {
                width: 100%;
                height: 500px;
                touch-action: none;
                background: gray;
            }
        </style>
    </head>
      )";
}

void log3d::canvas() {
  f << R"(
    <canvas id="renderCanvas"></canvas>
    <script>
        var canvas = document.getElementById("renderCanvas");
        var engine = new BABYLON.Engine(canvas, true);

        var createScene = function () {
            var scene = new BABYLON.Scene(engine);
            var color = new BABYLON.Color3(1, 1, 1);
            var material = new BABYLON.StandardMaterial("material", scene);
	          material.diffuseColor = color;

        )";

  f << R"(
      //Create a light
      var light = new BABYLON.DirectionalLight("Dir0", new BABYLON.Vector3(0, 0, 1), scene);
      light.diffuse = new BABYLON.Color3(1, 1, 1);
      light.specular = new BABYLON.Color3(1, 1, 1);
    )";
}

void log3d::ArcRotateCamera() {
  f << R"(
      //Create an Arc Rotate Camera - aimed negative z this time
      var camera = new BABYLON.ArcRotateCamera("Camera", Math.PI / 2, 1.0, -10, BABYLON.Vector3.Zero(), scene);

        	light.parent = camera;

          camera.inertia = 0;
	    camera.angularSensibilityX = 100;
	    camera.angularSensibilityY = 100;

          camera.attachControl(canvas, true);
)";
}

void log3d::axes() {
  f << R"(
	// Make origin sphere
	var origin_radius = 0.07;
	var origin = BABYLON.Mesh.CreateSphere("origin", 2, origin_radius, scene);
	origin.position = BABYLON.Vector3.Zero(); 

              // Make axes
	var xaxis = BABYLON.MeshBuilder.CreateCylinder("xaxis", { diameterBottom: origin_radius, diameterTop: origin_radius * 0.7, tessellation: 4, height: 1 }, scene);
	xaxis.rotation.z = Math.PI / 2;
	xaxis.position = new BABYLON.Vector3(.5, 0, 0);
	var yaxis = BABYLON.MeshBuilder.CreateCylinder("yaxis", { diameterBottom: origin_radius, diameterTop: origin_radius * 0.7, tessellation: 4, height:1 }, scene);
	yaxis.position = new BABYLON.Vector3(0, .5, 0);
	var zaxis = BABYLON.MeshBuilder.CreateCylinder("zaxis", { diameterBottom: origin_radius, diameterTop: origin_radius * 0.7, tessellation: 4, height:1 }, scene);
	zaxis.rotation.x = Math.PI / 2;
	zaxis.position = new BABYLON.Vector3(0, 0, .5);

          	var origin_material = new BABYLON.StandardMaterial("origin_material", scene);
	origin_material.diffuseColor = new BABYLON.Color3(1, 1, 1); //Black
	origin.material = origin_material;
	var xaxis_material = new BABYLON.StandardMaterial("xaxis_material", scene);
	xaxis_material.diffuseColor = new BABYLON.Color3(1, 0, 0); //Red
	xaxis.material = xaxis_material;
	var yaxis_material = new BABYLON.StandardMaterial("yaxis_material", scene);
	yaxis_material.diffuseColor = new BABYLON.Color3(0, 1, 0); //Green
	yaxis.material = yaxis_material;
	var zaxis_material = new BABYLON.StandardMaterial("zaxis_material", scene);
	zaxis_material.diffuseColor = new BABYLON.Color3(0, 0, 1); //Blue
	zaxis.material = zaxis_material;

                	// Creation of a lines mesh
	var lines = BABYLON.Mesh.CreateLines("lines", [
        new BABYLON.Vector3(1, 0, 0),
        new BABYLON.Vector3(0, 0, 0),
        new BABYLON.Vector3(0, 1, 0),
        new BABYLON.Vector3(0, 0, 0),
        new BABYLON.Vector3(0, 0, 1)
	], scene);

    )";
}

log3d& log3d::operator<<(std::string const& s) {
  f << s;
  return *this;
}

log3d::object_t log3d::newobj(std::string prefix)
{
  char buf[1024];
  sprintf_s(buf, "_%d", next_obj++);
  return prefix + buf;
}

log3d::object_t log3d::CreateSphere(int subdivisions, double radius) {
  object_t name = newobj("sphere");
  f << "  var " << name << " = BABYLON.Mesh.CreateSphere(\"" << name << "\", " << subdivisions << ", " << radius << ", scene);\n";
  f << "  " << name << ".material = material;\n";
  return name;
}

void log3d::position(object_t obj, double x, double y, double z) {
  f << "  " << obj << ".position.x = " << x << ";\n";
  f << "  " << obj << ".position.y = " << y << ";\n";
  f << "  " << obj << ".position.z = " << z << ";\n";
}

void log3d::rotation(object_t obj, double x, double y, double z) {
  f << "  " << obj << ".rotation.x = " << x << ";\n";
  f << "  " << obj << ".rotation.y = " << y << ";\n";
  f << "  " << obj << ".rotation.z = " << z << ";\n";
}

log3d::object_t log3d::CreatePlane(double v) {
  //Creation of a plane
  object_t name = newobj("sphere");
  f << "  var " << name << " = BABYLON.Mesh.CreatePlane(" << name << ", " << v << ", scene);\n";
  return name;
}
/*
//Creation of a material with wireFrame
var materialSphere1 = new BABYLON.StandardMaterial("texture1", scene);
materialSphere1.wireframe = true;

//Creation of a red material with alpha
var materialSphere2 = new BABYLON.StandardMaterial("texture2", scene);
materialSphere2.diffuseColor = new BABYLON.Color3(1, 0, 0); //Red

//Creation of a material with an image texture
var materialSphere3 = new BABYLON.StandardMaterial("texture3", scene);
materialSphere3.diffuseTexture = new BABYLON.Texture("textures/misc.jpg", scene);

//Creation of a material with translated texture
var materialSphere4 = new BABYLON.StandardMaterial("texture4", scene);
materialSphere4.diffuseTexture = new BABYLON.Texture("textures/misc.jpg", scene);
materialSphere4.diffuseTexture.vOffset = 0.1;//Vertical offset of 10%
materialSphere4.diffuseTexture.uOffset = 0.4;//Horizontal offset of 40%

//Creation of a material with an alpha texture
var materialSphere5 = new BABYLON.StandardMaterial("texture5", scene);
materialSphere5.diffuseTexture = new BABYLON.Texture("textures/tree.png", scene);
materialSphere5.diffuseTexture.hasAlpha = true;//Has an alpha

//Creation of a material and show all the faces
var materialSphere6 = new BABYLON.StandardMaterial("texture6", scene);
materialSphere6.diffuseTexture = new BABYLON.Texture("textures/tree.png", scene);
materialSphere6.diffuseTexture.hasAlpha = true;//Have an alpha
materialSphere6.backFaceCulling = false;//Show all the faces of the element

//Creation of a repeated textured material
var materialPlane = new BABYLON.StandardMaterial("texturePlane", scene);
materialPlane.diffuseTexture = new BABYLON.Texture("textures/grass.jpg", scene);
materialPlane.diffuseTexture.uScale = 5.0;//Repeat 5 times on the Vertical Axes
materialPlane.diffuseTexture.vScale = 5.0;//Repeat 5 times on the Horizontal Axes
materialPlane.backFaceCulling = false;//Always show the front and the back of an element

//Apply the materials to meshes
sphere1.material = materialSphere1;
sphere2.material = materialSphere2;

sphere3.material = materialSphere3;
sphere4.material = materialSphere4;

sphere5.material = materialSphere5;
sphere6.material = materialSphere6;

plane.material = materialPlane;

return scene;
};
*/

void log3d::mesh(Eigen::Matrix3Xi const& triangle_indices, Matrix3X const& V)
{
  int n = int(V.cols());

  f << "{    var positions = [\n";
  for (int i = 0; i < n; i++)
    f << "  " << V(0, i) << ", " << V(1, i) << ", " << V(2, i) << ", \n";
  f << "    ];\n";

  f << "    var indices = [\n";
  for (int face = 0; face < triangle_indices.cols(); face++)
    f << triangle_indices(0, face) << ", " << triangle_indices(1, face) << ", " << triangle_indices(2, face) << ",\n";
  f << "    ];\n";

  f << R"(
    var normals = [];
    BABYLON.VertexData.ComputeNormals(positions, indices, normals);

    var vertexData = new BABYLON.VertexData();
    vertexData.positions = positions;
    vertexData.indices = indices;
    vertexData.normals = normals;
    //vertexData.uvs = uvs;

    var polygon = new BABYLON.Mesh(name, scene);
    vertexData.applyToMesh(polygon);
    polygon.material = material;
  }
          )";
}


void log3d::wiremesh(Eigen::MatrixXi const& faces, Matrix3X const& V)
{
  int n = int(V.cols());
  object_t obj = newobj("wiremesh");
  f << "{\n";
  for (int face = 0; face < faces.cols(); ++face) {
    f << "/*var lines = */BABYLON.Mesh.CreateLines(\"" << obj << "_" << face << "\", [";
    for (int j = 0; j <= faces.rows(); j++) {
      int j0 = faces(j % faces.rows(), face);
      f << "  new BABYLON.Vector3(" << V(0, j0) << ", " << V(1, j0) << ", " << V(2, j0) << "),\n";
    }
    f << "], scene).color = color;\n";
  }
  f << "}\n";
}

void log3d::lines(Matrix3X const& V, bool closed)
{
  int n = int(V.cols());
  f << "/*var lines = */BABYLON.Mesh.CreateLines(\"" << newobj("lines") << "\", [";
  for (int j = 0; j <= n - (closed ? 1 : 0); j++) {
    int i = j % n;
    f << "  new BABYLON.Vector3(" << V(0, i) << ", " << V(1, i) << ", " << V(2, i) << "),\n";
  }
  f << "], scene).color = color;\n";
}

void log3d::star(Vector3 const & X)
{
  Scalar delta = 0.01f;
  Matrix3X pts(3, 2);
  pts.col(0) = X;
  pts.col(1) = X;
  pts(0, 0) -= delta;
  pts(0, 1) += delta;
  lines(pts);
  pts.col(0) = X;
  pts.col(1) = X;
  pts(1, 0) -= delta;
  pts(1, 1) += delta;
  lines(pts);
  pts.col(0) = X;
  pts.col(1) = X;
  pts(2, 0) -= delta;
  pts(2, 1) += delta;
  lines(pts);
}

void log3d::endcanvas() {
  f << R"(
      return scene;
    } // end createScene

        var scene = createScene();

        engine.runRenderLoop(function() {
      scene.render();
    });

        // Resize
    window.addEventListener("resize", function() {
      engine.resize();
    });
    </script>
)";
}


void log3d::done() {
  f << R"(
      </body>
      </html>
)";
}


