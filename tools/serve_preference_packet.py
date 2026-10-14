"""Serve one explicit local AV annotation packet, never the repository/SSD."""
import argparse,json,os
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);parser.add_argument('--device',type=int,required=True)
    args=parser.parse_args();root=args.packet.resolve(strict=True)
    if not root.is_relative_to(Path('/Volumes/CrossCutMedia/CrossCut-Validation')) or root.stat().st_dev!=args.device:
        raise ValueError('explicit authorized SSD packet required')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['contract']!='unlabelled_same_event_av_triplets_v1':raise ValueError('wrong packet')
    allowed={'/','/index.html','/manifest.json'}
    for row in manifest['events']:
        for name in row['clips'].values():
            if Path(name).name!=name or not (root/name).is_file() or (root/name).is_symlink():
                raise ValueError('only named regular packet movies may be served')
            allowed.add('/'+name)
    template=Path(__file__).parent/'templates/same_event_preference_review.html';target=root/'index.html'
    if target.exists() and target.read_bytes()!=template.read_bytes():raise FileExistsError('existing review page differs; do not overwrite')
    if not target.exists():target.write_bytes(template.read_bytes())
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if not Path('/Volumes/CrossCutMedia').is_mount() or root.stat().st_dev!=args.device:
                self.send_error(503,'Authorized SSD disconnected');return
            if unquote(urlsplit(self.path).path) not in allowed:
                self.send_error(404);return
            super().do_GET()
        do_HEAD=do_GET
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(root)))
    print(json.dumps(dict(url=f'http://127.0.0.1:{server.server_port}/',pid=os.getpid(),
        packet=root.name,allowed_paths=len(allowed),external_network_exposure=False)),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
