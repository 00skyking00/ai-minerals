<?php
// 1x1-pixel visit logger for johnsondevco.com/ai-minerals/.
//
// The CSV lives one directory above the ai-minerals web root so the
// deploy script's `rsync --delete-excluded` against ${REMOTE_DIR}/
// does NOT wipe accumulated visit data on every push. Layout on
// Hostinger after first request:
//
//   public_html/
//     ai-minerals/
//       beacon.php          <- this file
//     _visits/
//       ai_minerals.csv     <- log
//       .htaccess           <- "Require all denied" so the CSV isn't web-readable
//
// Each row is comma-quoted UTF-8:
//   iso8601_time, remote_ip, referer, user_agent, accept_language

$log_dir  = __DIR__ . '/../_visits';
$log_file = $log_dir . '/ai_minerals.csv';
$htaccess = $log_dir . '/.htaccess';

if (!is_dir($log_dir)) {
    @mkdir($log_dir, 0755, true);
}
if (!file_exists($htaccess)) {
    @file_put_contents(
        $htaccess,
        "# Deny direct web access to the visit log.\n" .
        "Require all denied\n" .
        "<IfModule !mod_authz_core.c>\n  Order deny,allow\n  Deny from all\n</IfModule>\n"
    );
}

$row = [
    date('c'),
    $_SERVER['REMOTE_ADDR']         ?? '',
    $_SERVER['HTTP_REFERER']        ?? '',
    $_SERVER['HTTP_USER_AGENT']     ?? '',
    $_SERVER['HTTP_ACCEPT_LANGUAGE'] ?? '',
];

$f = @fopen($log_file, 'a');
if ($f !== false) {
    @flock($f, LOCK_EX);
    fputcsv($f, $row);
    @flock($f, LOCK_UN);
    fclose($f);
}

// 1x1 transparent GIF89a.
header('Content-Type: image/gif');
header('Cache-Control: no-store, no-cache, must-revalidate, max-age=0');
header('Pragma: no-cache');
header('Content-Length: 43');
echo base64_decode('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7');
