/**
 * Custom getFilesFromEvent for react-dropzone that supports folder drops.
 * When a folder is dragged onto the dropzone, this recursively reads all
 * files inside it and returns them as a flat File[] array.
 */

const ACCEPTED_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp',
  '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv',
  '.mp3', '.wav',
  '.txt',
]);

function getExtension(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot >= 0 ? name.slice(dot).toLowerCase() : '';
}

function readEntriesPromise(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((resolve, reject) => {
    reader.readEntries(resolve, reject);
  });
}

async function readAllEntries(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  const entries: FileSystemEntry[] = [];
  // readEntries returns batches (max ~100), so we must loop until empty
  let batch: FileSystemEntry[];
  do {
    batch = await readEntriesPromise(reader);
    entries.push(...batch);
  } while (batch.length > 0);
  return entries;
}

function fileEntryToFile(entry: FileSystemFileEntry): Promise<File> {
  return new Promise((resolve, reject) => {
    entry.file(resolve, reject);
  });
}

async function collectFiles(entry: FileSystemEntry): Promise<File[]> {
  if (entry.isFile) {
    const ext = getExtension(entry.name);
    if (!ACCEPTED_EXTENSIONS.has(ext)) return [];
    // skip hidden files
    if (entry.name.startsWith('.')) return [];
    try {
      const file = await fileEntryToFile(entry as FileSystemFileEntry);
      return [file];
    } catch {
      return [];
    }
  }

  if (entry.isDirectory) {
    // skip hidden directories and cache directories
    if (entry.name.startsWith('.') || entry.name.startsWith('_')) return [];
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    const entries = await readAllEntries(reader);
    const nested = await Promise.all(entries.map(e => collectFiles(e)));
    return nested.flat();
  }

  return [];
}

export async function getFilesFromEvent(
  event: React.DragEvent | React.ChangeEvent<HTMLInputElement> | Event,
): Promise<File[]> {
  // Handle input change events (click to select) - pass through
  if ('target' in event && (event as any).target?.files) {
    return Array.from((event as any).target.files);
  }

  // Handle drag events
  const dragEvent = event as DragEvent;
  if (!dragEvent.dataTransfer) return [];

  const items = dragEvent.dataTransfer.items;
  if (!items || items.length === 0) {
    // Fallback: use files directly
    return Array.from(dragEvent.dataTransfer.files);
  }

  // Check if any items are directories using webkitGetAsEntry
  const entries: FileSystemEntry[] = [];
  const plainFiles: File[] = [];

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const entry = item.webkitGetAsEntry?.();
    if (entry) {
      entries.push(entry);
    } else if (item.kind === 'file') {
      const file = item.getAsFile();
      if (file) plainFiles.push(file);
    }
  }

  if (entries.length > 0) {
    const nested = await Promise.all(entries.map(e => collectFiles(e)));
    return nested.flat();
  }

  // Fallback: filter plain files by extension
  return plainFiles.filter(f => ACCEPTED_EXTENSIONS.has(getExtension(f.name)));
}
