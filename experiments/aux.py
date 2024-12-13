import heapq
import torch

def loss_analysis(config, loss_list, k=5):
    k_largest = heapq.nlargest(k, enumerate(loss_list), key=lambda x: x[1])
    k_smallest = heapq.nsmallest(k, enumerate(loss_list), key=lambda x: x[1])
    
    indices_largest, values_largest = zip(*k_largest)
    indices_smallest, values_smallest = zip(*k_smallest)
    
    values_largest = [round(value, 4) for value in values_largest]
    values_smallest = [round(value, 4) for value in values_smallest]
    
    return indices_largest, values_largest, indices_smallest, values_smallest

def station_wave_information(config, data, verbose=False):
    station_wave_dict = {}
    
    for station in data:
        station_tensor = station['y_test']
    
        max_wave = torch.max(station_tensor)
        min_wave = torch.min(station_tensor)
        avg_wave = torch.mean(station_tensor)
        var_wave = torch.var(station_tensor)
    
        if verbose:
            print()
            print(f'Station: {station["name"]}')
            print(f'Max Wave Height: {max_wave.item():.2f} meters')
            print(f'Min Wave Height: {min_wave.item():.2f} meters')
            print(f'Average Wave Height: {avg_wave.item():.2f} meters')
            print(f'Variance Wave Height: {var_wave.item():.2f} meters')
            print()
        
        station_wave_dict[station['name']] = {
            'max_wave': max_wave.item(),
            'min_wave': min_wave.item(),
            'avg_wave': avg_wave.item(),
            'var_wave': var_wave.item()
        }
    
    return station_wave_dict

def station_wave_information_ndbc(config, data, verbose=False):
    
    max_wave_train = torch.max(data['y_train'])
    max_wave_test = torch.max(data['y_test'])
    min_wave_train = torch.min(data['y_train'])
    min_wave_test = torch.min(data['y_test'])
    
    avg_wave_train = torch.mean(data['y_train'])
    avg_wave_test = torch.mean(data['y_test'])
    
    var_wave_train = torch.var(data['y_train'])
    var_wave_test = torch.var(data['y_test'])

    if verbose:
        print()
        print(f'Max Wave Height: {max_wave_train.item():.2f}, {max_wave_test.item():.2f}')
        print(f'Min Wave Height: {min_wave_train.item():.2f}, {min_wave_test.item():.2f}')
        print(f'Average Wave Height: {avg_wave_train.item():.2f}, {avg_wave_test.item():.2f}')
        print(f'Variance Wave Height: {var_wave_train.item():.2f}, {var_wave_test.item():.2f}')
        print()