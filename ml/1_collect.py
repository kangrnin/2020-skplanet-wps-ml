import subprocess
import time
import datetime
import pandas as pd
from pathlib import Path

def get_wifis():
    subprocess.run(
        ['netsh', 'interface', 'set', 'interface', 'name="Wi-Fi"', 'admin=disabled'],
        capture_output=True)
    subprocess.run(['netsh', 'interface', 'set', 'interface', 'name="Wi-Fi"', 'admin=enabled'],
        capture_output=True)

    time.sleep(3)
    output = subprocess.run(
        ['netsh', 'wlan', 'show', 'network', 'mode=Bssid'],
        capture_output=True, text=True, encoding='ISO-8859-1').stdout

    results = output.split('\n\n')[1:-1]

    timestamp = datetime.datetime.now()
    wifis = []
    for result in results:
        lines = result.split('\n')

        for i in range(len(lines)):
            if lines[i].split()[0] == 'BSSID':
                bssid = lines[i].split()[-1]
                rssi = int(lines[i+1].split()[-1][:-1])
                if rssi > 50:
                    wifis.append({'bssid':bssid, 'rssi':rssi, 'timestamp':timestamp})

    return wifis

def collect(position, rp):
    data_path = Path('../wifi_data') / position
    data_path.mkdir(parents=True, exist_ok=True)

    wifi_list = list()
    bssid_set = set()
    overlap_cnt = 0
    scan_cnt = 0

    while overlap_cnt < 3:
        scan_wifis = get_wifis()

        new_bssid = [ w['bssid'] for w in scan_wifis if w['bssid'] not in bssid_set ]
        bssid_set.update(new_bssid)
        if new_bssid:
            wifi_list += scan_wifis
            overlap_cnt = 0
        else:
            overlap_cnt += 1

        scan_cnt += 1

        print('-----------------------------------------')
        print(f'scan #{ scan_cnt } total bssid cnt : {len(scan_wifis)}')
        print(f'{ len(new_bssid) } new bssids : ')
        print(', '.join([f'~{ b[-8:] }' for b in new_bssid]))

    print('scan complete. bssid cnt : '+str(len(bssid_set)))

    df = pd.DataFrame(wifi_list)
    df['position'] = position
    df['rp'] = rp

    df.to_csv(data_path / (f'{ rp }.csv'), mode='a', index=False, header=False)
    df.to_csv(data_path / 'wifi_all.csv', mode='a', index=False, header=False)

if __name__ == "__main__":
    position = input('Enter Position : ')
    rp = input('Enter RP : ')
    collect(position, rp)
        
        