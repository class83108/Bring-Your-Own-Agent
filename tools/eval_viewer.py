"""Eval Conversation Viewer — 獨立 Web 伺服器。

用法：
    python tools/eval_viewer.py [--port 8501] [--db eval-results/eval.db]

開啟瀏覽器 http://localhost:8501 即可瀏覽 eval 對話歷史。
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import webbrowser
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).resolve().parent


class EvalViewerHandler(SimpleHTTPRequestHandler):
    """處理 API 請求與靜態頁面。"""

    db_path: str = ''

    def do_GET(self) -> None:  # noqa: N802
        """路由 GET 請求。"""
        if self.path == '/' or self.path == '/index.html':
            self._serve_html()
        elif self.path == '/api/runs':
            self._api_runs()
        elif self.path.startswith('/api/runs/'):
            run_id = self.path.split('/api/runs/')[1].rstrip('/')
            self._api_run_detail(run_id)
        elif self.path.startswith('/api/results/') and self.path.endswith('/conversation'):
            result_id = self.path.split('/api/results/')[1].replace('/conversation', '')
            self._api_conversation(result_id)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_html(self) -> None:
        """提供 HTML 頁面。"""
        html_path = TOOLS_DIR / 'eval_viewer.html'
        content = html_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json_response(self, data: object) -> None:
        """回傳 JSON 回應。"""
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_conn(self) -> sqlite3.Connection:
        """取得資料庫連線。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _api_runs(self) -> None:
        """GET /api/runs — 列出所有 eval runs。"""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT r.run_id, r.agent_version, r.model, r.created_at, r.notes,
                       COUNT(e.id) as total_tasks,
                       SUM(e.passed) as passed,
                       ROUND(AVG(e.score), 2) as avg_score,
                       SUM(e.total_tokens) as total_tokens
                FROM eval_runs r
                LEFT JOIN eval_results e ON r.run_id = e.run_id
                GROUP BY r.run_id
                ORDER BY r.created_at DESC
            """).fetchall()
            self._json_response([dict(r) for r in rows])
        finally:
            conn.close()

    def _api_run_detail(self, run_id: str) -> None:
        """GET /api/runs/{run_id} — 取得 run 的所有 task 結果。"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, task_name, task_level, passed, score,
                       details, tool_calls, total_tokens,
                       duration_seconds, ran_verification, error
                FROM eval_results
                WHERE run_id = ?
                ORDER BY task_name
                """,
                (run_id,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                # 將 details JSON 字串解析為物件
                if d.get('details'):
                    d['details'] = json.loads(d['details'])
                results.append(d)
            self._json_response(results)
        finally:
            conn.close()

    def _api_conversation(self, result_id: str) -> None:
        """GET /api/results/{result_id}/conversation — 取得對話歷史。"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                'SELECT conversation, task_name FROM eval_results WHERE id = ?',
                (result_id,),
            ).fetchone()
            if row is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            conversation = json.loads(row['conversation']) if row['conversation'] else []
            self._json_response(
                {
                    'task_name': row['task_name'],
                    'conversation': conversation,
                }
            )
        finally:
            conn.close()

    def log_message(self, format: str, *args: object) -> None:
        """靜音 HTTP 請求日誌，僅保留 API 呼叫。"""
        if '/api/' in str(args[0]) if args else False:
            super().log_message(format, *args)


def main() -> None:
    """啟動 Eval Viewer 伺服器。"""
    parser = argparse.ArgumentParser(description='Eval Conversation Viewer')
    parser.add_argument('--port', type=int, default=8501, help='伺服器埠號 (預設: 8501)')
    parser.add_argument('--db', type=str, default='eval-results/eval.db', help='SQLite 資料庫路徑')
    parser.add_argument('--no-open', action='store_true', help='不自動開啟瀏覽器')
    args = parser.parse_args()

    if not Path(args.db).exists():
        logger.error('找不到資料庫: %s', args.db)
        return

    EvalViewerHandler.db_path = args.db

    HTTPServer.allow_reuse_address = True
    server = HTTPServer(('localhost', args.port), EvalViewerHandler)
    url = f'http://localhost:{args.port}'
    print(f'Eval Viewer 啟動於 {url}')  # noqa: T201
    print(f'資料庫: {args.db}')  # noqa: T201
    print('按 Ctrl+C 停止')  # noqa: T201

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止')  # noqa: T201
        server.server_close()


if __name__ == '__main__':
    main()
