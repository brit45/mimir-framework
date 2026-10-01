#!/usr/bin/env python3
"""Check a Viz host under Xvfb; no training or dataset is needed."""
import subprocess, os, time, urllib.request, urllib.error, re, struct
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description='Graphical Viz integration checks; run under xvfb-run. Requires Pillow and xdotool for desktop hosts.')
parser.add_argument('--backend', required=True, choices=['SFML', 'QT', 'GTK', 'WEB'])
parser.add_argument('--executable', required=True)
parser.add_argument('--host', help='Optional path to mimir_viz_host')
parser.add_argument('--output', required=True)
parser.add_argument('--xdotool', default='xdotool')
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
E = dict(os.environ, LIBGL_ALWAYS_SOFTWARE='1', QT_QPA_PLATFORM='xcb', GDK_BACKEND='x11')
xd = args.xdotool

def run(args):
    return subprocess.check_output([xd] + args, env=E, text=True).strip()

def start(name, exe, host=None):
    env = E.copy()
    if host:
        env['MIMIR_VIZ_HOST'] = host
    log = open(str(out / ('viz-' + name + '.log')), 'w')
    p = subprocess.Popen([exe, '20', str(out / ('viz-' + name + '.png'))], env=env, stdout=log, stderr=log)
    return (p, log)
for name, exe, host in [] if args.backend == 'WEB' else [(args.backend, args.executable, args.host)]:
    p, log = start(name, exe, host)
    try:
        for i in range(60):
            time.sleep(0.1)
            if 'PIXELS_OK' in Path(str(out / ('viz-' + name + '.log'))).read_text():
                break
            if p.poll() is not None:
                raise AssertionError(Path(str(out / ('viz-' + name + '.log'))).read_text())
        wid = run(['search', '--name', '^Mimir Viz backend smoke$']).splitlines()[-1]
        run(['windowfocus', '--sync', wid])
        run(['mousemove', '--window', wid, '20', '30'])
        run(['click', '1'])
        run(['click', '4'])
        run(['key', 'h'])
        run(['key', 'ctrl+shift+Left'])
        run(['key', 'r'])
        time.sleep(0.5)
        geom = run(['getwindowgeometry', '--shell', wid])
        assert 'WIDTH=400' in geom and 'HEIGHT=280' in geom, geom
        from PIL import ImageGrab
        screen = ImageGrab.grab(xdisplay=E.get('DISPLAY'))
        screen.save(str(out / ('viz-' + name + '-host.png')))
        g = dict((line.split('=', 1) for line in geom.splitlines()))
        x, y = (int(g['X']), int(g['Y']))
        assert screen.getpixel((x + 20, y + 30))[:3] == (231, 42, 63), (name, screen.getpixel((x + 20, y + 30)))
        import ctypes
        X = ctypes.CDLL('libX11.so.6')
        X.XOpenDisplay.restype = ctypes.c_void_p
        d = X.XOpenDisplay(E['DISPLAY'].encode())
        X.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        X.XInternAtom.restype = ctypes.c_ulong

        class Data(ctypes.Union):
            _fields_ = [('b', ctypes.c_char * 20), ('s', ctypes.c_short * 10), ('l', ctypes.c_long * 5)]

        class Client(ctypes.Structure):
            _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int), ('display', ctypes.c_void_p), ('window', ctypes.c_ulong), ('message_type', ctypes.c_ulong), ('format', ctypes.c_int), ('data', Data)]

        class XEvent(ctypes.Union):
            _fields_ = [('client', Client), ('pad', ctypes.c_long * 24)]
        event = XEvent()
        event.client.type = 33
        event.client.display = d
        event.client.window = int(wid)
        event.client.message_type = X.XInternAtom(d, b'WM_PROTOCOLS', 0)
        event.client.format = 32
        event.client.data.l[0] = X.XInternAtom(d, b'WM_DELETE_WINDOW', 0)
        X.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long, ctypes.POINTER(XEvent)]
        X.XSendEvent(d, int(wid), 0, 0, ctypes.byref(event))
        X.XFlush.argtypes = [ctypes.c_void_p]
        X.XFlush(d)
        X.XCloseDisplay.argtypes = [ctypes.c_void_p]
        X.XCloseDisplay(d)
        assert p.wait(timeout=5) == 0
        text = Path(str(out / ('viz-' + name + '.log'))).read_text()
        for token in ['PIXELS_OK', 'DOWN 20 30', 'UP 20 30', 'WHEEL 1', 'TEXT 104', 'CLOSE']:
            assert token in text, (name, token, text)
        assert re.search('KEY \\d+ 1 1', text), text
        print(name, 'pixels, host pixels, keyboard, modifiers, click, wheel, resize, close OK', flush=True)
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait()
        log.close()
if args.backend != 'WEB':
    raise SystemExit(0)
p, log = start('WEB', args.executable)
try:
    for i in range(60):
        time.sleep(0.1)
        text = Path(str(out / 'viz-WEB.log')).read_text()
        m = re.search('http://127.0.0.1:\\d+/[a-f0-9]+/', text)
        if m and 'PIXELS_OK' in text:
            break
    assert m, text
    base = m.group(0).rstrip('/')

    def req(path, method='GET'):
        return urllib.request.urlopen(urllib.request.Request(base + path, method=method), timeout=3)
    assert b'<canvas' in req('/').read()
    bmp = req('/frame').read()
    assert bmp[:2] == b'BM'
    assert struct.unpack_from('<ii', bmp, 18) == (320, -240)
    assert bmp[54 + (30 * 320 + 20) * 4:54 + (30 * 320 + 20) * 4 + 3] == bytes([63, 42, 231])
    for t, v in [(3, 0), (4, 0), (5, 0), (6, 72), (8, 104), (7, 72), (6, 82)]:
        assert req(f'/event?t={t}&x=20&y=30&v={v}&m=3&d=1', 'POST').status == 204
    time.sleep(0.4)
    assert struct.unpack_from('<ii', req('/frame').read(), 18) == (400, -280)
    for path, status, method in [('/event?t=bad', 400, 'POST'), ('/event?t=6&x=999999999999999999999&y=0&v=1&m=0&d=0', 400, 'POST'), ('/event?t=1&x=0&y=0&v=0&m=0&d=0', 404, 'GET')]:
        try:
            req(path, method)
            raise AssertionError('accepted invalid request')
        except urllib.error.HTTPError as e:
            assert e.code == status
    try:
        urllib.request.urlopen(base.rsplit('/', 1)[0] + '/wrong/frame')
        raise AssertionError('accepted wrong token')
    except urllib.error.HTTPError as e:
        assert e.code == 404
    req('/event?t=1&x=0&y=0&v=0&m=0&d=0', 'POST')
    assert p.wait(timeout=5) == 0
    text = Path(str(out / 'viz-WEB.log')).read_text()
    for token in ['PIXELS_OK', 'DOWN 20 30', 'UP 20 30', 'WHEEL 1', 'TEXT 104', 'CLOSE']:
        assert token in text, (token, text)
    print('WEB pixels, HTTP, events, resize, invalid requests, close OK', flush=True)
finally:
    if p.poll() is None:
        p.terminate()
        p.wait()
    log.close()
