#!/usr/bin/env python3
"""Isolated HTTP integration test; no access to the user's dataset."""
import base64
import copy
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import urllib.parse

APP = Path(__file__).resolve().parents[1]
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jGZkAAAAASUVORK5CYII=')


class Client:
    def __init__(self, base):
        self.base = base
        self.root = None
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        page = self.opener.open(base + '/').read().decode()
        self.token = re.search(r'name="csrf-token" content="([^"]+)"', page)[1]

    def post(self, endpoint, payload, expected=200, token=None):
        if endpoint not in ['scan', 'create']:
            payload = {**payload, 'dataset_root': self.root}
        req = urllib.request.Request(self.base + '/api/' + endpoint + '.php', json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json', 'X-CSRF-Token': self.token if token is None else token})
        try:
            response = self.opener.open(req)
        except urllib.error.HTTPError as exc:
            response = exc
        raw = response.read()
        try:
            result = json.loads(raw)
        except ValueError:
            raise AssertionError((endpoint, response.code, raw))
        assert response.code == expected, (endpoint, response.code, result)
        if endpoint == 'scan' and response.code == 200:
            self.root = result['root']
        return result

    def upload(self, root, relative_path, content, expected=200, token=None):
        boundary = 'lumen-test-boundary'
        fields = [('dataset_root', str(root)), ('relative_path', relative_path)]
        chunks = []
        for name, value in fields:
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        filename = relative_path.rsplit('/', 1)[-1]
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
        chunks.append(content + f'\r\n--{boundary}--\r\n'.encode())
        req = urllib.request.Request(self.base + '/api/upload.php', b''.join(chunks), {
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'X-CSRF-Token': self.token if token is None else token})
        try:
            response = self.opener.open(req)
        except urllib.error.HTTPError as exc:
            response = exc
        result = json.loads(response.read())
        assert response.code == expected, ('upload', response.code, result)
        return result


def main():
    with tempfile.TemporaryDirectory(prefix='lumen-annotation-test-') as temp:
        root = Path(temp) / 'dataset'
        (root / 'portraits').mkdir(parents=True)
        (root / 'animals').mkdir()
        for name in ['portraits/étoile.PNG', 'portraits/second.png', 'animals/dog.png']:
            (root / name).write_bytes(PNG)
        originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.png')}
        (Path(temp) / 'outside.png').write_bytes(PNG)
        (root / 'escape.png').symlink_to(Path(temp) / 'outside.png')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        sessions = Path(temp) / 'sessions'; sessions.mkdir()
        cmd = [os.environ.get('PHP_BIN', 'php'), *shlex.split(os.environ.get('PHP_ARGS', '')), '-d', 'session.save_path=' + str(sessions), '-S', f'127.0.0.1:{port}', '-t', str(APP)]
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        client = Client(base)
                        break
                    except urllib.error.URLError:
                        if process.poll() is not None:
                            log.seek(0)
                            raise RuntimeError(log.read().decode())
                        time.sleep(.05)
                else:
                    raise RuntimeError('PHP server unavailable')
                directories = client.post('directories', {'path':temp})
                assert directories['path'] == str(Path(temp).resolve())
                assert client.post('directories', {'path':temp}, expected=403, token='invalid')
                created = client.post('create', {'parent':temp,'name':'dataset-importé'})
                imported_root = Path(created['root'])
                assert imported_root.is_dir() and imported_root.parent == Path(temp)
                imported = client.upload(imported_root, 'chapitre 1/page.png', PNG)
                assert imported == {'relative_path':'chapitre 1/page.png','skipped':False}
                assert client.upload(imported_root, 'chapitre 1/page.png', PNG)['skipped']
                composer_sidecar = {'normalized':True,'boxes':[{'x1':.1,'y1':.2,'x2':.8,'y2':.9,'class_id':0,'label':'Masse'}]}
                client.upload(imported_root, 'chapitre 1/page.json', json.dumps(composer_sidecar).encode())
                composition = {'type':'medical_detector_dataset','class_names':['Masse'],'images':1}
                client.upload(imported_root, 'composition.mimir-meta', json.dumps(composition).encode())
                client.upload(imported_root, 'chapitre 1/masks/page_mask.png', PNG)
                imported_scan = client.post('scan', {'root':str(imported_root)})
                assert imported_scan['images'] == ['chapitre 1/page.png'] and imported_scan['annotated'] == []
                assert imported_scan['mask_links'] == {'chapitre 1/page.png':['chapitre 1/masks/page_mask.png']}
                converted = client.post('load', {'image':'chapitre 1/page.png'})
                converted_annotation = converted['annotation']
                assert converted_annotation['captions']['primary'] == 'Image annotée : Masse.'
                assert converted_annotation['entities'][0]['phrase'] == 'Masse'
                assert converted_annotation['entities'][0]['bbox'] == [.1,.2,.8,.9]
                converted_annotation['captions']['primary'] = 'Mammographie présentant une masse.'
                saved_composer = client.post('save', {'image':'chapitre 1/page.png','annotation':converted_annotation,
                                                       'revision':converted['revision']})
                stored_composer = json.loads((imported_root/'chapitre 1/page.json').read_text())
                assert stored_composer['boxes'] == [{'x1':.1,'y1':.2,'x2':.8,'y2':.9,'class_id':0,'label':'Masse'}]
                assert stored_composer['captions']['primary'] == 'Mammographie présentant une masse.'
                assert (imported_root/'composition.mimir-meta').is_file()
                client.post('scan', {'root':str(imported_root)})
                client.upload(imported_root, 'chapitre 1/page.png', PNG+b'changed', 409)
                client.upload(imported_root, '../outside.png', PNG, 403)
                client.upload(imported_root, 'notes.txt', PNG, 400)
                client.upload(imported_root, 'auth.png', PNG, 403, token='invalid')
                client.post('create', {'parent':temp,'name':'dataset-importé'}, expected=409)
                client.post('create', {'parent':temp,'name':'../escape'}, expected=400)
                imported_scan = client.post('scan', {'root':str(imported_root)})
                assert imported_scan['images'] == ['chapitre 1/page.png']
                assert client.post('scan', {'root':str(root)})['images'] == ['animals/dog.png','portraits/second.png','portraits/étoile.PNG']
                scanned = client.post('scan', {'root': str(root)})
                assert len(scanned['images']) == 3 and len(scanned['warnings']) == 1
                broken_image = root / 'animals/broken.png'
                broken_sidecar = root / 'animals/broken.json'
                broken_image.write_bytes(PNG)
                broken_sidecar.write_text('{"invalid":true}')
                rescanned = client.post('scan', {'root':str(root)})
                assert 'animals/broken.png' not in rescanned['images']
                assert any('animals/broken.png' in warning and 'image ignorée' in warning for warning in rescanned['warnings'])
                broken_image.unlink()
                broken_sidecar.unlink()
                scanned = client.post('scan', {'root':str(root)})
                assert len(scanned['images']) == 3
                relative = 'portraits/étoile.PNG'
                loaded = client.post('load', {'image': relative})
                assert loaded['revision'] is None and not loaded['exists']
                a = loaded['annotation']
                a['captions']['primary'] = 'Une femme près d’une fenêtre, éclairée par le soleil.'
                a['captions']['alternatives'] = ['Un portrait baigné de lumière.']
                a['entities'] = [{'id':'person_1','phrase':'une femme','role':'primary_subject','bbox':[.1,.2,.5,.9],
                                  'mask':{'type':'polygon','points':[[.1,.2],[.5,.2],[.5,.9],[.1,.9]]},
                                  'track':{'id':'chapter-asha','name':'Asha'}},
                                 {'id':'window_1','phrase':'une fenêtre','role':'object','bbox':[.6,0,1,1]}]
                a['entities'].append({'id':'floor_1','phrase':'sol en pierre','role':'background',
                                      'mask':{'type':'polygon','points':[[0,.75],[1,.75],[1,1],[0,1]]}})
                a['entities'].append({'id':'text_1','phrase':'texte dans une bulle','role':'text',
                                      'mask':{'type':'polygon','points':[[.2,.1],[.5,.1],[.5,.2],[.2,.2]]},
                                      'ocr':{'text':'こんにちは','translation':'Bonjour','source_language':'ja','target_language':'fr','tags':['dialogue','salutation']}})
                a['attributes'] = [{'entity':'person_1','property':'hair_color','value':'black'}]
                a['relations'] = [{'subject':'person_1','predicate':'near','object':'window_1'},
                                  {'subject':'person_1','predicate':'on','object':'floor_1'}]
                a['custom_metadata'] = {'preserved': True}
                a['entities'][0]['skeleton'] = {'name':'humain', 'points':[
                    {'name':'hanche','position':[.3,.6],'parent':None,'role':'articulation majeure'},
                    {'name':'tête','position':[.3,.2],'parent':'hanche','role':'tête'},
                    {'name':'bras gauche','position':None,'parent':'hanche'}]}

                a['entities'][0]['skeleton']['points'][1]['entities'] = ['person_1','window_1','floor_1']
                a['entities'][0]['skeleton']['edges'] = [{'from':'tête','to':'bras gauche','label':'articulation'}]
                saved = client.post('save', {'image':relative, 'annotation':a, 'revision':None})
                assert (root / 'portraits/étoile.json').exists()
                assert json.loads((root / 'portraits/étoile.json').read_text()) == a
                reloaded = client.post('load', {'image':relative})
                assert reloaded['annotation'] == a and reloaded['revision'] == saved['revision']
                client.post('save', {'image':relative,'annotation':a,'revision':None}, 409)
                for mutation in ['position', 'duplicate', 'parent', 'name', 'shape', 'role']:
                    bad = copy.deepcopy(a)
                    skeleton = bad['entities'][0]['skeleton']
                    if mutation == 'position': skeleton['points'][0]['position'] = [1.1, .5]
                    if mutation == 'duplicate': skeleton['points'][1]['name'] = 'hanche'
                    if mutation == 'parent': skeleton['points'][1]['parent'] = 'absent'
                    if mutation == 'name': skeleton['name'] = ' '
                    if mutation == 'role': skeleton['points'][0]['role'] = 'invalid'
                    if mutation == 'shape': skeleton['points'][0]['position'] = [.5]
                    client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                for mutation in ['unknown_entity','duplicate_entity','bad_edge','self_edge']:
                    bad = copy.deepcopy(a); skeleton = bad['entities'][0]['skeleton']
                    if mutation == 'unknown_entity': skeleton['points'][1]['entities'] = ['missing']
                    if mutation == 'duplicate_entity': skeleton['points'][1]['entities'] = ['person_1','person_1']
                    if mutation == 'bad_edge': skeleton['edges'][0]['to'] = 'missing'
                    if mutation == 'self_edge': skeleton['edges'][0]['to'] = 'tête'
                    client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                for mutation in ['mask_type','mask_point','mask_area']:
                    bad=copy.deepcopy(a)
                    if mutation=='mask_type':bad['entities'][2]['mask']['type']='bitmap'
                    if mutation=='mask_point':bad['entities'][2]['mask']['points'][0]=[-.1,.75]
                    if mutation=='mask_area':bad['entities'][2]['mask']['points']=[[0,.5],[.5,.5],[1,.5]]
                    client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                for mutation in ['track_without_mask','track_without_name','ocr_without_mask','ocr_empty_text','ocr_bad_tag']:
                    bad=copy.deepcopy(a)
                    if mutation=='track_without_mask':bad['entities'][0].pop('mask')
                    if mutation=='track_without_name':bad['entities'][0]['track']['name']=' '
                    if mutation=='ocr_without_mask':bad['entities'][3].pop('mask')
                    if mutation=='ocr_empty_text':bad['entities'][3]['ocr']['text']=' '
                    if mutation=='ocr_bad_tag':bad['entities'][3]['ocr']['tags']=['dialogue',' ']
                    client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                legacy = copy.deepcopy(a); del legacy['entities'][0]['skeleton']
                legacy_saved = client.post('save', {'image':relative,'annotation':legacy,'revision':saved['revision']})
                assert client.post('load', {'image':relative})['annotation'] == legacy
                saved = client.post('save', {'image':relative,'annotation':a,'revision':legacy_saved['revision']})

                bad = copy.deepcopy(a); bad['entities'][0]['bbox'] = [-.1,0,1,1]
                client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                bad = copy.deepcopy(a); bad['relations'][0]['object'] = 'missing'
                client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                bad = copy.deepcopy(a); bad['captions']['primary'] = ' '
                client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                client.post('load', {'image':'../outside.png'}, 403)
                client.post('load', {'image':'escape.png'}, 404)
                client.post('state', {}, 403, token='invalid')
                image = client.opener.open(base + '/api/image.php?image=portraits/second.png&dataset_root=' + urllib.parse.quote(str(root)))
                assert image.headers['Content-Type'] == 'image/png' and image.read() == PNG
                client.post('state', {'image':'portraits/second.png'})
                state = json.loads((root / '.annotation_state.json').read_text())
                assert state['last_image'] == 'portraits/second.png' and relative in state['annotated']
                fresh = Client(base)
                resumed = fresh.post('scan', {'root':str(root)})
                assert resumed['images'][resumed['resume_index']] == 'portraits/second.png'
                assert resumed['state']['last_annotation'] == relative and resumed['annotated'] == [relative]
                fresh.post('state', {'image':relative})
                resumed = Client(base).post('scan', {'root':str(root)})
                assert resumed['images'][resumed['resume_index']] not in resumed['annotated']
                assert resumed['identities']==[{'id':'chapter-asha','name':'Asha','images':[relative],'conflicting_names':[]}]
                broken = Path(temp) / 'broken'; broken.mkdir(); (broken / '.annotation_state.json').write_text('invalid')
                fresh.post('scan', {'root':str(broken)}, 400)
                assert fresh.post('load', {'image':relative})['annotation'] == a
                without_track = copy.deepcopy(a)
                del without_track['entities'][0]['track']
                removed_track = fresh.post('save', {'image':relative,'annotation':without_track,'revision':saved['revision']})
                refreshed = fresh.post('scan', {'root':str(root)})
                assert refreshed['identities'] == []
                saved = removed_track
                a = without_track
                # A conflicting existing JSON must never be replaced with a blank annotation.
                unrelated = root / 'animals/dog.json'; unrelated.write_text('{"foreign":true}')
                fresh.post('load', {'image':'animals/dog.png'}, 400)
                assert unrelated.read_text() == '{"foreign":true}'
                unrelated.unlink()
                (root / 'portraits/second.jpg').write_bytes(PNG)
                fresh.post('load', {'image':'portraits/second.png'}, 409)
                # A sidecar symlink cannot overwrite anything outside the selected root.
                target = Path(temp) / 'outside.json'; target.write_text('{"untouched":true}')
                (root / 'animals/dog.json').symlink_to(target)
                fresh.post('save', {'image':'animals/dog.png','annotation':a,'revision':None}, 409)
                assert target.read_text() == '{"untouched":true}'
                req = urllib.request.Request(base + '/', headers={'Host':'attacker.example'})
                try:
                    fresh.opener.open(req)
                    raise AssertionError('Foreign Host accepted')
                except urllib.error.HTTPError as exc:
                    assert exc.code == 403
                other = Path(temp) / 'other'; other.mkdir(); (other / 'dog.png').write_bytes(PNG)
                client.post('scan', {'root':str(other)})
                client.root = str(root)  # Simulate the stale tab sharing this session.
                client.post('load', {'image':relative}, 409)
                for path, digest in originals.items():
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                assert not list(root.rglob('.annotation-tmp-*'))
                print('PASS: dataset creation, image import/retry safety, recursive scan, UTF-8 sidecars, OCR, tracked identities, reload, polygon masks, metadata, normalized boxes, references, revisions, exact resume, CSRF, Host, traversal, symlinks, name collisions, scan rollback, dataset switch, unchanged images.')
            except Exception:
                log.seek(0)
                print(log.read().decode())
                raise
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == '__main__':
    main()
