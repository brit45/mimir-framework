<?php
require dirname(__DIR__) . '/config.php';
api(function ($d) {
    $path = realpath($d['path'] ?? '');
    if (!$path || !is_dir($path) || !is_readable($path)) fail('Dossier inaccessible.');
    $directories = [];
    foreach (new DirectoryIterator($path) as $entry) {
        if ($entry->isDot() || !$entry->isDir() || $entry->isLink() || !is_readable($entry->getPathname())) continue;
        $directories[] = ['name'=>$entry->getFilename(),'path'=>$entry->getPathname()];
    }
    usort($directories, static fn($a,$b) => strnatcasecmp($a['name'],$b['name']));
    $parent = dirname($path);
    return ['path'=>$path,'parent'=>$parent === $path ? null : $parent,'directories'=>$directories];
});
