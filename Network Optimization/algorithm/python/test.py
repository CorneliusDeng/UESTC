import os

# read port file
def read_port_file(filename):
    ports = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # ignore the first line
        for line in lines[1:]:
            port_id, bandwidth = map(int, line.strip().split(','))
            # store port id and bandwidth in ports
            ports[port_id] = bandwidth
    return ports

# read flow file
def read_flow_file(filename):
    flows = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # ignore the first line
        for line in lines[1:]:
            flow_id, bandwidth, enter_time, send_time = map(int, line.strip().split(','))
            # store flow id and bandwidth, enter time and send time in flows
            flows[flow_id] = {
                'bandwidth': bandwidth,
                'enter_time': enter_time,
                'send_time': send_time
            }
    return flows

def greedy_algorithm(ports, flows):
    result = []
    
    return result


def main():
    # set the directory path
    dir_path = '../data'
    # use os.listdir() to get all files and folders in the directory
    files_and_folders = os.listdir(dir_path)
    
    folder_names = []
    for item in files_and_folders:
        item_path = os.path.join(dir_path, item)  
        # judge whether it is a folder
        if os.path.isdir(item_path): 
            folder_names.append(item)

    # read port file and flow file in each folder
    for folder in folder_names:
        ports = read_port_file(f'{dir_path}/{folder}/port.txt')
        flows = read_flow_file(f'{dir_path}/{folder}/flow.txt')
        result = greedy_algorithm(ports, flows)
        # write result to result.txt
        with open(f'{dir_path}/{folder}/result.txt', 'w', encoding='utf-8') as f:
            for r in result:
                f.write(f'{r[0]},{r[1]},{r[2]}\r\n')

if __name__ == '__main__':
    main()