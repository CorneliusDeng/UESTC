import csv

'''
总的算法步骤是:
1、读入输入数据
2、对流量进行优先级排序,查找最佳分配端口
3、记录并更新分配结果
4、将分配结果写出

代码解析：
1、读入端口带宽数据和流量数据,存入ports和flows两个字典中。ports是端口id和带宽对照表,flows是流量数据列表。
2、遍历每个流量,查找有足够剩余带宽的端口,分配该流量。先按照剩余带宽排序,再按进入设备时间+发送流所需时间排序,以实现最优分配。
3、分配结果存入result列表中,包含流id、分配的端口id和开始发送时间。
4、将result结果写出到文件result.txt中。
'''

# 读取端口文件数据
def read_ports(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # 跳读第一行头部信息
        next(reader)  
        # 读取端口id和带宽列表
        ports = {int(row[0]): int(row[1]) for row in reader}
    return ports

# 读取流量文件数据
def read_flows(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # 跳读第一行头部信息
        next(reader)
        # 读取流量数据列表
        flows = [{'id': int(row[0]), 'bandwidth': int(row[1]), 'entry_time': int(row[2]), 'duration': int(row[3])} for row in reader]
    return flows

# 分配流量
def allocate_flows(ports, flows):
    result = []
    sum_flow = 0
    seen_entries = set()
    free_port_ids = []
    num_virtual_ports = 5

    for flow in flows: 
        # 搜索可用端口 
        available_ports = [(port_id, port_bandwidth - flow['bandwidth']) for port_id, port_bandwidth in ports.items() if port_id not in seen_entries] 
        available_ports += [(x, ports[x]) for x in free_port_ids] 
        # 按带宽和端口号排序 
        available_ports = sorted(available_ports, key=lambda x: x[1], reverse=True) 
        # 按进入设备时间+发送流所需时间从小到大排列
        available_ports = sorted(available_ports, key=lambda y: y[0]*flow['entry_time'] + y[0]*flow['duration'])
        for port_id, available_bandwidth in available_ports: 
            # 确保有足够带宽 
            if available_bandwidth >= 0: 
                # 确保不早于入口时间1个单位 
                start_time = max(flow['entry_time'], 1) 
                # 添加分配结果 
                result.append({'flow_id': flow['id'], 'port_id': port_id, 'start_time': start_time}) 
                sum_flow += flow['bandwidth'] 
                seen_entries.add(port_id) 
                # 更新端口带宽 
                ports[port_id] -= flow['bandwidth'] 
                if available_bandwidth == 0: free_port_ids.append(port_id) 
                break 
    return result, sum_flow

def write_result(filename, result):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['流id', '端口id', '开始发送时间'])
         # 写入所有分配结果行
        for entry in result:
            writer.writerow([entry['flow_id'], entry['port_id'], entry['start_time']])

def main():
    ports = read_ports('../data/port.txt')
    flows = read_flows('../data/flow.txt')
    result, sum_flow = allocate_flows(ports, flows)
    print('Flow send result: ', sum_flow)
    write_result('../data/result.txt', result)

if __name__ == '__main__':
    main()