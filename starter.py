import os
import uuid
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
import sys, platform
import tkinter as tk
import tkinter.messagebox 
import psutil
import GPUtil
import json
import requests
from webdriver_manager.chrome import ChromeDriverManager
import systemInfo
import multiprocessing as mp
import threading
import ctypes
from collections import defaultdict


cfg = None
binary_location = ''
chrome: webdriver.Chrome = None
monitor_proc: mp.Process = None
monitored_system_state = []
URL = 'http://10.172.138.88'
MEASUREMENT_PORT = 13366
WASM_URL_PREFIX = ""
INCLUDE_QOE = False
msg_que: mp.Queue
stop_flag: bool = False
log_flag: str = None
QoE_webpage = f"https://www.youtube.com/watch?v=8NK9FJYixiI"

    

def _monitor(pid, que: mp.Queue):
    # print("monitor process")
    DISCONNECTED_MSG = 'Unable to evaluate script: disconnected: not connected to DevTools\n'
    INTERVAL = 0.1
    idx = 1
    chrome_proc = psutil.Process(pid)
    children_proc = chrome_proc.children(recursive=True)
    tab_procs = [_p for _p in children_proc if '--type=renderer' in _p.cmdline()]
    tab_pids = [proc.pid for proc in tab_procs]
    que.put(True)
    print("begin monitoring CPU utilization")
    while True:
        # if idx % 5 == 0:
        #     sys.stdout.write("\033[F")
        #     sys.stdout.write("\033[K")
        #     print("monitoring CPU utilization" + "." * (idx // 5 % 6 + 1), end="\r", flush=True)
        # print(idx, [psutil.pid_exists(pid) for pid in tab_pids])
        try:
            if not all([psutil.pid_exists(pid) for pid in tab_pids]):
            # if chrome.get_log('driver')[-1]['message'] == DISCONNECTED_MSG:
                print('Browser or Tab closed by user')
                break
            else:
                # print("shit", idx, proc.status())
                _tmp = {
                    "cpu_percent": [proc.cpu_percent(interval=INTERVAL) for proc in tab_procs],
                    "virtual_memory": [proc.memory_info() for proc in tab_procs],
                    "timestamp": time.time() * 1000,
                }
                que.put(_tmp)
                idx += 1
            # time.sleep(1)
        except IndexError: 
            pass
        except Exception:
            break


def _listen():
    global msg_que, monitor_proc, chrome, monitored_system_state

    p = psutil.Process(chrome.service.process.pid)
    # print(p.pid, p.cmdline(), '\n')
    # for _child in p.children(recursive=True):
    #     print(_child.pid, _child.cmdline())
    # print(chrome.service.process.pid, p.children(recursive=True), p.children())
    msg_que = mp.Queue()
    print("monitoring...")
    monitor_proc = mp.Process(target=_monitor, args=(p.pid, msg_que), daemon=True)
    # print("web tab pid", p.children()[0].pid)
    monitor_proc.start()
    while not stop_flag:
        if not msg_que.empty():
            msg = msg_que.get()
            if isinstance(msg, bool):
                web_btn = chrome.find_element(value="button")
                web_btn.click()
            else:
                monitored_system_state.append(msg_que.get())
    print("stop monitoring")
    monitor_proc.terminate()



def start_btn_click():
    global chrome, stop_flag, window
    stop_flag = True

    if chrome:
        print("\n\n")
        chrome = None
        
    try:
        print("begin downloading chromedriver......")
        d = DesiredCapabilities.CHROME
        d['goog:loggingPrefs'] = { 'browser':'ALL' }
        # print("trying newing webdriver via Service")
        chrome = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_opt, desired_capabilities=d)
    except Exception as e:
        tkinter.messagebox.showerror('error', 'error, close the app please')
        window.destroy()
        exit()

    print("chromedriver downloaded, start chrome and get system profile......")
    sys_info = {
        "os": platform.platform(),
        "cpu": systemInfo.GetCpuConstants(),
        "processor": platform.processor(),
        "chrome": chrome.capabilities['browserVersion'],
        "memory": round(psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024.0), 2),
        "igpu": platform.processor() if not GPUtil.getGPUs() else None,
        "dgpu": GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else None,
    }
    if sys.platform == "darwin":
        sys_info["cpu_model"] = os.popen('sysctl -n machdep.cpu.brand_string').read().strip()
    try:
        print("uploading system profile...")
        requests.post(url=f'{URL}:{MEASUREMENT_PORT}/data/hardware', headers={'Content-Type': 'application/json'}, data=json.dumps(sys_info))
        print("system profile uploaded")
    except requests.exceptions.ConnectionError:
        print("fail to upload")
    print(f"try opening {URL}:{MEASUREMENT_PORT}/index.html ......")
    chrome.get(f"{URL}:{MEASUREMENT_PORT}/index.html")    
    stop_flag = False
    listening_t = threading.Thread(target=_listen, daemon=True)
    listening_t.start()


def QoE_btn_click():
    global chrome, stop_flag, window
    stop_flag = True

    if chrome:
        print("\n\n")
        chrome = None
    try:
        print("downloading chromedriver......")
        d = DesiredCapabilities.CHROME
        d['goog:loggingPrefs'] = { 'browser':'ALL' }
        # print("trying newing webdriver via Service")
        chrome = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_opt, desired_capabilities=d)
    except Exception as e:
        tkinter.messagebox.showerror('error', 'error, please close app')
        window.destroy()
        exit()
    
    
    chrome.get(QoE_webpage)
    print("finish get population website")
    chrome.execute_script("if (typeof define != \"undefined\" && define.amd) {delete define.amd;}")
    js = ""
    for script_fn in ["dist/tf-core.js", "dist/tf-backend-cpu.js", 
                        "dist/tf-backend-webgl.js", "dist/tf-backend-wasm.min.js", 
                        "dist/tf-converter.js", "dist/ort.min.js", "index.js"]:
        # response = requests.get(f'{URL}:{MEASUREMENT_PORT}/{script_fn}')
        # js += "\n" + response.content.decode("utf-8") + "\n"
        with open(f'./{script_fn}') as f:
            js += "\n" + f.read() + "\n"
    if "speedometer" in QoE_webpage:
        js += " \
            async function startTestAsync() { \
                startTest(); \
            }   \
            entry_func(\"" + WASM_URL_PREFIX + "\");\
            startTestAsync(); \
        "
    else:
        js += f"\nentry_func(\"{WASM_URL_PREFIX}\")"
        
    print(len(js))
    time.sleep(10)
    print("start inference")
    chrome.execute_script(js)




def on_closing():
    global chrome, window, monitored_system_state, stop_flag, monitor_proc
    
    remove_useless_state = False
    if not chrome:
        stop_flag = True
        window.destroy()
    
    stop_flag = True
    # print(globals())
    if "msg_que" not in globals() or not msg_que or (msg_que.empty() and not monitored_system_state):
        print("just return")
        if monitor_proc:
            monitor_proc.terminate()
        window.destroy()
        return
    # todo: send hardware usage data.
    try:
        while not msg_que.empty():
            monitored_system_state.append(msg_que.get())
        
        # 2 options: determined by Analyser.
        #   - remove useless states from the beginning and the end of state list and then linear map time axis 
        #   - collect exact length state that has same time period length with inference process.
        if remove_useless_state:
            while not any(monitored_system_state[0]):
                monitored_system_state.pop(0)
            while not any(monitored_system_state[-1]):
                monitored_system_state.pop(-1)

        # print("len(monitored_system_state)", len(monitored_system_state))
        all_log = chrome.get_log('browser')
        # print(len(all_log))
        # print(*[c['message'][:200] for c in all_log], sep='\n')
        tfjs_wasm_profiling_jsonstr, tfjs_webgl_profiling_jsonstr = "", ""

        ort_wasm_log_dict = defaultdict(dict)
        ort_webgl_log_dict = defaultdict(dict)
        ort_cur_model = None
        ort_webgl_model_list = []
        new_model_log_flag = True
        ort_webgl_cur_model_index = 0
        ort_webgl_mem_cache = []
        inference_timestamp = {
            "tfjs": {
                "wasm": {},
                "webgl": {}
            },
            "ort": {
                "wasm": {},
                "webgl": {}
            },
        }

        for c in all_log:
            line = c["message"]
            try:
                if "TFJS_PROFILING_RESULT_WASM" in line:
                    tfjs_wasm_profiling_jsonstr = json.loads(line.split('"TFJS_PROFILING_RESULT_WASM" ')[1])
                    # print("get tfjs_wasm_profiling_jsonstr", len(json.loads(tfjs_wasm_profiling_jsonstr)))
                
                elif "TFJS_PROFILING_RESULT_WEBGL" in line:
                    tfjs_webgl_profiling_jsonstr = json.loads(line.split('"TFJS_PROFILING_RESULT_WEBGL" ')[1])
                    # print("get tfjs_webgl_profiling_jsonstr", len(json.loads(tfjs_webgl_profiling_jsonstr)))
                
                if "ORT_BEGIN_INFERENCE:wasm" in line:
                    ort_cur_model = line.split('"ORT_BEGIN_INFERENCE:wasm"')[1].strip().strip('"')
                    # print(f'ort begin inference wasm {ort_cur_model}')
                elif "logEventInmediately" in line:
                    if "kernels" not in ort_wasm_log_dict[ort_cur_model]:
                        ort_wasm_log_dict[ort_cur_model]["kernels"] = []
                    _ort_wasm_kernel_jsonstr = json.loads('"' + line.split("logEventInmediately")[1].strip())
                    # print(type(_ort_wasm_kernel_jsonstr), _ort_wasm_kernel_jsonstr)
                    ort_wasm_log_dict[ort_cur_model]["kernels"].append(json.loads(_ort_wasm_kernel_jsonstr))
                elif "ORT_BEGIN_INFERENCE:webgl" in line or "ORT_FINISH_INFERENCE:webgl" in line:
                    _model = line.split(':webgl"')[1].strip().split()[0].strip('"')
                    if _model not in ort_webgl_model_list:
                        ort_webgl_model_list.append(_model)
                    new_model_log_flag = True

                elif "Profiler.op" in line:
                    if new_model_log_flag:
                        new_model_log_flag = False
                        ort_cur_model = ort_webgl_model_list[ort_webgl_cur_model_index]
                        ort_webgl_cur_model_index += 1
                    if "kernels" not in ort_webgl_log_dict[ort_cur_model]:
                        ort_webgl_log_dict[ort_cur_model]["kernels"] = []
                    ort_webgl_log_dict[ort_cur_model]["kernels"].append(line.split("|")[1].strip().strip('"'))

                elif "ORT_INFERENCE_BEGIN_MEMORY:wasm" in line:
                    ort_wasm_log_dict[ort_cur_model]["inference_begin_memory"] = json.loads(json.loads(line.split('"ORT_INFERENCE_BEGIN_MEMORY:wasm"')[1].strip()))
                elif "ORT_INFERENCE_SESSION_MEMORY:wasm" in line:
                    ort_wasm_log_dict[ort_cur_model]["session_setup_memory"] = json.loads(json.loads(line.split('"ORT_INFERENCE_SESSION_MEMORY:wasm"')[1].strip()))
                elif "ORT_INFERENCE_FINISH_MEMORY:wasm" in line:
                    ort_wasm_log_dict[ort_cur_model]["inference_finish_memory"] = json.loads(json.loads(line.split('"ORT_INFERENCE_FINISH_MEMORY:wasm"')[1].strip()))
                elif "ORT_INFERENCE_BEGIN_MEMORY:webgl" in line:
                    ort_webgl_mem_cache.append(["inference_begin_memory", json.loads(json.loads(line.split('"ORT_INFERENCE_BEGIN_MEMORY:webgl"')[1].strip()))])
                elif "ORT_INFERENCE_SESSION_MEMORY:webgl" in line:
                    ort_webgl_mem_cache.append(["session_setup_memory", json.loads(json.loads(line.split('"ORT_INFERENCE_SESSION_MEMORY:webgl"')[1].strip()))])
                elif "ORT_INFERENCE_FINISH_MEMORY:webgl" in line:
                    ort_webgl_mem_cache.append(["inference_finish_memory", json.loads(json.loads(line.split('"ORT_INFERENCE_FINISH_MEMORY:webgl"')[1].strip()))])
            
                elif "ORT_INFERENCE_TIMESTAME" in line:
                    # print(line, line.strip().split()[-1].strip('"').split(':'))
                    backend, model, timestamp = line.strip().split()[-1].strip('"').split(':')
                    inference_timestamp["ort"][backend][model] = timestamp
                elif "TFJS_INFERENCE_TIMESTAME" in line:
                    # print(line, line.strip().split()[-1].strip('"').split(':'))
                    backend, model, timestamp = line.strip().split()[-1].strip('"').split(':')
                    inference_timestamp["tfjs"][backend][model] = timestamp

            except json.decoder.JSONDecodeError:
                print("fail to decode line:", line[:200])
        
        try:
            requests.post(url=f'{URL}:{MEASUREMENT_PORT}/data/ort/wasm-log', headers={'Content-Type': 'application/json'}, data=json.dumps(ort_wasm_log_dict))
        except requests.exceptions.ConnectionError:
            print("fail to upload ort_wasm_log_dict")
        try:
            requests.post(url=f'{URL}:{MEASUREMENT_PORT}/data/ort/webgl-log', headers={'Content-Type': 'application/json'}, data=json.dumps(ort_webgl_log_dict))
        except requests.exceptions.ConnectionError:
            print("fail to upload ort_webgl_log_dict ")
        try:
            requests.post(url=f'{URL}:{MEASUREMENT_PORT}/data/monitor', headers={'Content-Type': 'application/json'}, data=json.dumps(monitored_system_state))
        except requests.exceptions.ConnectionError:
            print("fail to upload monitored_system_state ")
        try:
            requests.post(url=f'{URL}:{MEASUREMENT_PORT}/data/timestamp', headers={'Content-Type': 'application/json'}, data=json.dumps(inference_timestamp))
        except requests.exceptions.ConnectionError:
            print("fail to upload inference_timestamp ")
        # for fn in ["tfjs_wasm_profiling_jsonstr", "tfjs_webgl_profiling_jsonstr", "ort_wasm_log_dict", "ort_webgl_log_dict", "monitored_system_state"]:
        #     with open(f'data/{fn}.json', 'w') as f:
        #         json.dump(eval(fn), f, indent=2)
        monitor_proc.terminate()
        window.destroy()
        print("finish all")
    except:
        window.destroy()




if __name__ == "__main__":
    mp.freeze_support()
    if sys.platform == "darwin":
        binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if not os.path.exists(binary_location):
            os.system("""osascript -e \'Tell application \"System Events\" to display dialog \"please install chrome\" with title \"error\"\'""")
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        elif __file__:
            application_path = os.path.dirname(__file__)
        else:
            application_path = ""
        
        if os.path.exists(os.path.join(application_path, "config/config.json")):
            with open(os.path.join(application_path, "config/config.json")) as f:
                cfg = json.load(f)
        else:
            print(os.path.join(application_path, "config/config.json"))

    elif sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\Clients\StartMenuInternet\Google Chrome\DefaultIcon")
            value, _ = winreg.QueryValueEx(key, "")
            binary_location = value.split(',')[0]
        except FileNotFoundError:
            ctypes.windll.user32.MessageBoxW(0, "please install chrome", "error", 1)
            sys.exit()
        if os.path.exists(os.path.join(os.getcwd(), "config/config.json")):
            with open(os.path.join(os.getcwd(), "config/config.json")) as f:
                cfg = json.load(f)
        else:
            print(os.path.abspath(os.path.join(os.getcwd(), "config/config.json")))
    
    if cfg:
        URL = cfg["URL"]
        MEASUREMENT_PORT = cfg["PORT"]
        WASM_URL_PREFIX = cfg["WASM_URL_PREFIX"]
        INCLUDE_QOE = cfg["INCLUDE_QOE"]
        QoE_webpage = cfg["QoE_webpage"]

    window = tk.Tk("Test")
    tk.Label(window, text='Chrome executable is here. Press [Start] and wait\nuntil there is alert box in chrome, close all', font=("Calibri 15")).pack(fill=tk.X,  padx=20, pady=5)
    en = tk.Entry(window, width=60, relief=tk.GROOVE, font=("Calibri 15"), justify=tk.CENTER)
    en.insert(0, binary_location)
    en.pack(fill=tk.X, padx=20, pady=10)
    start_btn = tk.Button(window, text="Start", font=("Calibri 15"), relief=tk.GROOVE, command=start_btn_click)
    start_btn.pack(fill=tk.X, padx=20, pady=5)
    if INCLUDE_QOE:
        QoE_btn = tk.Button(window, text="WebQoE", font=("Calibri 15"), relief=tk.GROOVE, command=QoE_btn_click)
        QoE_btn.pack(fill=tk.X, padx=20, pady=5)

    chrome_opt = webdriver.ChromeOptions()
    chrome_opt.add_argument("--disable-web-security")
    chrome_opt.add_argument("--enable-unsafe-webgpu")
    chrome_opt.add_argument('--no-sandbox')
    chrome_opt.add_argument('--ignore-certificate-errors')
    chrome_opt.add_argument("disable-dawn-features=disallow_unsafe_apis")
    chrome_opt.add_argument("enable-features=SharedArrayBuffer")

    
    window.protocol("WM_DELETE_WINDOW", on_closing)
    try:
        window.mainloop()
    except:
        window.destroy()






# p = psutil.Process(chrome.service.process.pid)
# try:
#     while chrome.current_url:
#         pass
# except Exception:
#     pass


