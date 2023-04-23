import os

# 读取端口文件，将端口的id和带宽存储在字典ports中，并返回ports字典
def read_port_file(filename):
    ports = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            port_id, bandwidth = map(int, line.strip().split(','))
            ports[port_id] = bandwidth
    return ports

# 读取流文件，将流的id、带宽、进入时间和发送时间存储在字典flows中，并返回flows字典
def read_flow_file(filename):
    flows = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            flow_id, bandwidth, enter_time, send_time = map(int, line.strip().split(','))
            flows[flow_id] = {
                'bandwidth': bandwidth,
                'enter_time': enter_time,
                'send_time': send_time
            }
    return flows

def greedy_algorithm(ports, flows):
    # 按流进入时间排序，确保调度顺序
    flows = sorted(flows.items(), key=lambda x: x[1]['enter_time'])
    
    # 初始化每个端口的状态，包括是否空闲，当前已发送流的带宽，当前已发送流的结束时间等信息
    port_state = {port_id: {'is_free': True, 'bandwidth': 0, 'end_time': 0} for port_id in ports}
    
    result = []
    for flow_id, flow_info in flows:
        # 计算每个端口上剩余带宽的最小值
        min_remaining_bandwidth = min(ports[port_id] - port_state[port_id]['bandwidth'] for port_id in ports)
        
        # 初始化选择的出端口和发送时间
        selected_port = -1
        selected_start_time = float('inf')
        
        # 在每个空闲的出端口上发送该流，并选择在该时间段内剩余带宽最大的端口
        for port_id in ports:
            if port_state[port_id]['is_free']:
                # 计算该流在该端口的发送时间
                start_time = max(port_state[port_id]['end_time'], flow_info['enter_time'])
                end_time = start_time + flow_info['send_time']
                # 如果该端口可以在该时间段内发送该流，并且在该时间段内剩余带宽最大，则选择该端口
                if end_time <= start_time + (min_remaining_bandwidth - port_state[port_id]['bandwidth']) / 2:
                    if port_state[port_id]['bandwidth'] + flow_info['bandwidth'] <= ports[port_id]:
                        if end_time < selected_start_time:
                            selected_port = port_id
                            selected_start_time = end_time
        
        # 如果没有空闲的出端口，则选择一个最早结束的出端口
        if selected_port == -1:
            for port_id in ports:
                if port_state[port_id]['end_time'] < selected_start_time:
                    selected_port = port_id
                    selected_start_time = port_state[port_id]['end_time']
        
        # 如果找不到可用的出端口，则提示错误
        if selected_port == -1:
            raise ValueError('Cannot schedule flow due to insufficient bandwidth')
        
        # 在选择的出端口上发送该流，并更新端口状态
        start_time = max(port_state[selected_port]['end_time'], flow_info['enter_time'])
        end_time = start_time + flow_info['send_time']
        port_state[selected_port]['bandwidth'] += flow_info['bandwidth']
        port_state[selected_port]['is_free'] = False
        port_state[selected_port]['end_time'] = end_time
        result.append((flow_id, selected_port, start_time))
    
    return result


def main():
    # 指定目录路径
    dir_path = '../data'
    # 使用os.listdir()列出所有文件和文件夹
    files_and_folders = os.listdir(dir_path)
    # 遍历每一个文件或文件夹，筛选出所有文件夹并返回它们的文件名
    folder_names = []
    for item in files_and_folders:
        item_path = os.path.join(dir_path, item)  # 构建文件或文件夹的完整路径
        if os.path.isdir(item_path):  # 判断是否是文件夹
            folder_names.append(item)

    for folder in folder_names:
        ports = read_port_file(f'{dir_path}/{folder}/port.txt')
        flows = read_flow_file(f'{dir_path}/{folder}/flow.txt')
        result = greedy_algorithm(ports, flows)

        with open(f'{dir_path}/{folder}/result.txt', 'w', encoding='utf-8') as f:
            for r in result:
                f.write(f'{r[0]},{r[1]},{r[2]}\r\n')

if __name__ == '__main__':
    main()