import trimesh
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Convert OBJ to GLB and DAE using Trimesh.')
    parser.add_argument('input_obj', type=str, help='Path to the input OBJ file')
    args = parser.parse_args()

    input_path = args.input_obj

    # Load the OBJ file (assumes .mtl and textures are in the same directory)
    mesh = trimesh.load(input_path, force='scene')  # scene preserves hierarchy and materials better

    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path) or "."

    # Export to GLB
    glb_path = os.path.join(output_dir, base_name + '.glb')
    mesh.export(file_type='glb', file_obj=glb_path)

    # # Export to DAE
    # dae_path = os.path.join(output_dir, base_name + '.dae')
    # mesh.export(file_type='dae', file_obj=dae_path)

    # print(f"Exported to:\n  {glb_path}\n  {dae_path}")


# python convert_obj.py path/to/model.obj
#  python obj_to_glb.py models/a_cups/textured.obj
if __name__ == '__main__':
    main()
