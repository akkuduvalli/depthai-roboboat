from depthai_sdk import OakCamera

# Download & deploy a model from Roboflow universe:
# # https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters/dataset/6

with OakCamera(usb_speed='usb2') as oak:
    color = oak.create_camera('color')
    model_config = {
        'source': 'roboflow', # Specify that we are downloading the model from Roboflow
        'model':'gnc-buoys-2024',
        'key':'cnKNvPkhl3HaFSuhYAiO' # Fake API key, replace with your own!
    }
    nn = oak.create_nn(model_config, color)
    oak.visualize(nn, fps=True)
    oak.start(blocking=True)