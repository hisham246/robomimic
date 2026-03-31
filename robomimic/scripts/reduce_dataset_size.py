import h5py
import argparse
import os

def cut_dataset(input_dataset_path, output_dataset_path, num_demos_to_keep):
    with h5py.File(input_dataset_path, 'r') as input_file:
        data_group = input_file['data']
        
        # Get only the demo keys (filters out any 'mask' keys if they exist)
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        selected_demos = demo_keys[:num_demos_to_keep]
        
        with h5py.File(output_dataset_path, 'w') as output_file:
            # 1. Copy file-level metadata (vital for Robomimic)
            for k, v in input_file.attrs.items():
                output_file.attrs[k] = v
                
            output_group = output_file.create_group('data')
            
            # 2. Copy data-group metadata (contains env_args)
            for k, v in data_group.attrs.items():
                output_group.attrs[k] = v
                
            total_frames = 0
            
            # 3. Copy the selected demos recursively using h5py's built-in copy
            for demo_key in selected_demos:
                # This recursively copies the demo, including the nested 'obs' group
                data_group.copy(demo_key, output_group)
                
                # Accumulate frames from each demo
                total_frames += output_group[demo_key].attrs.get('num_samples', 0)
                
            # 4. Update the total_frames attribute so the new dataset loads correctly
            if 'total_frames' in output_group.attrs:
                output_group.attrs['total_frames'] = total_frames
                
            print(f"Dataset has been cut to {len(selected_demos)} demos and saved to {output_dataset_path}")

def main():
    parser = argparse.ArgumentParser(description="Cut an HDF5 dataset to a specific number of demos.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input HDF5 dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to the output HDF5 dataset')
    parser.add_argument('--num_demos', type=int, required=True, help='Number of demos to keep in the output dataset')

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cut_dataset(args.input, args.output, args.num_demos)

if __name__ == "__main__":
    main()