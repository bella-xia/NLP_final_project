from gpt2_finetune import read_json
import json, os, numpy

def combine_jsons(root_dir, path_list, output_path):
    json_full_data = []
    for path in path_list:
        json_data  = read_json(os.path.join(root_dir, path))
        processed_json_data = [modify_json_content(instance) for instance in json_data]
        print(processed_json_data[-1])
        json_full_data.extend(processed_json_data)
    json_string = json.dumps(json_full_data, indent=4)
    with open(os.path.join(root_dir, output_path), "w") as json_file:
        json_file.write(json_string)

def modify_json_content(data):
    instruction = data['input']
    encoded_io = data['temp 1.5 top_k 30']
    partitioned_encoded_io = encoded_io[len(instruction):]
    while partitioned_encoded_io[0] == ' ' or partitioned_encoded_io[0] == '.':
        partitioned_encoded_io = partitioned_encoded_io[1:]
    return {"instruction": instruction,
    "output": partitioned_encoded_io}

if __name__ == '__main__':                                   
    path_list = numpy.arange(4) + 1
    path_list = [f"forward_generation_epoch_{path_num}.json" for path_num in path_list]
    print(path_list)
    combine_jsons('/home/zxia15/NLP_final_project/executables', path_list, 'forward_generation_full.json')