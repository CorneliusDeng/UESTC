import pulp
import os

# 读取端口文件，将端口的id和带宽存储在字典ports中，并返回ports字典
def read_port_file(filename):
    ports = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            port_id, bandwidth = map(int, line.strip().split(','))
            ports[port_id] = bandwidth
    return ports

# 读取流文件，将流的id、带宽、进入时间和发送时间存储在字典flows中，并返回flows字典
def read_flow_file(filename):
    flows = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            flow_id, bandwidth, enter_time, send_time = map(int, line.strip().split(','))
            flows[flow_id] = {
                'bandwidth': bandwidth,
                'enter_time': enter_time,
                'send_time': send_time
            }
    return flows

def linear_programming(ports, flows):
    # 创建线性规划模型
    model = pulp.LpProblem('Flow_Scheduling', pulp.LpMinimize)
    # 每个端口的带宽使用量，整数变量
    ports_vars = {}
    # 每个流分配到的端口，二进制变量
    flows_vars = {}
    for port_id in ports:
        ports_vars[port_id] = pulp.LpVariable(f'port_{port_id}', lowBound=0, cat='Integer')
    for flow_id in flows:
        flows_vars[flow_id] = {}
        for port_id in ports:
            flows_vars[flow_id][port_id] = pulp.LpVariable(f'flow_{flow_id}_port_{port_id}', lowBound=0, cat='Binary')

    # 约束条件1：每个端口的带宽使用量不得超过其总带宽
    for port_id in ports:
        # ports[port_id]表示端口port_id的总带宽，ports_vars[port_id]表示端口port_id的带宽使用量
        model += sum(flows_vars[flow_id][port_id] * flows[flow_id]['bandwidth'] for flow_id in flows) <= ports[port_id] * ports_vars[port_id]
    
    # 约束条件2：每个流只能分配到一个端口，并且流进入时间不得早于分配到的端口的发送时间
    for flow_id in flows:
        # 每个流只能分配到一个端口
        model += sum(flows_vars[flow_id][port_id] for port_id in ports) == 1
        for port_id in ports:
            # 流进入时间不得早于分配到的端口的发送时间
            model += flows_vars[flow_id][port_id] * flows[flow_id]['send_time'] + ports_vars[port_id] >= flows[flow_id]['enter_time']

    # 将所有端口的带宽使用量相加，作为目标函数
    model += sum(ports_vars.values())

    # 求解线性规划模型
    model.solve()

    # 获取每个流分配到的端口、开始发送时间和剩余可用带宽，并将结果存储在result列表中
    result = []
    for flow_id in flows:
        for port_id in ports:
            if flows_vars[flow_id][port_id].value() == 1:
                result.append((flow_id, port_id, int(pulp.value(ports_vars[port_id]) - flows[flow_id]['send_time'])))
                break
    # 按流id对结果进行排序
    result = sorted(result, key=lambda x: x[0])
    return result

def main():
     # 指定目录路径
    dir_path = '../data'

    # 使用os.listdir()列出所有文件和文件夹
    files_and_folders = os.listdir(dir_path)

    # 遍历每一个文件或文件夹，统计文件夹数量
    num_folders = 0
    for item in files_and_folders:
        item_path = os.path.join(dir_path, item)  # 构建文件或文件夹的完整路径
        if os.path.isdir(item_path):  # 判断是否是文件夹
            num_folders += 1

    for i in range(0,num_folders):
        ports = read_port_file(f'../data/{i}/port.txt')
        flows = read_flow_file(f'../data//{i}/flow.txt')

        result = linear_programming(ports, flows)

        with open(f'../data/{i}/result.txt', 'w', encoding='utf-8') as f:
            for r in result:
                f.write(f'{r[0]},{r[1]},{r[2]}\r\n')

if __name__ == '__main__':
    main()