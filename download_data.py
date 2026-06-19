

from roboflow import Roboflow
rf = Roboflow(api_key="EWzq7imfM4uyrdi23ft7")
project = rf.workspace("leaf-gfzji").project("plantvillage-6yjer")
version = project.version(1)
dataset = version.download("folder")
                