'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { ChevronLeft, ChevronRight, X } from 'lucide-react';
import { apiClient } from '@/utils/api';
import { isAudio, isVideo } from '@/utils/basic';

interface Props {
  imgPath: string | null;
  images: string[];
  onChange: (nextPath: string | null) => void;
}

function fileName(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

export default function DatasetImageViewer({ imgPath, images, onChange }: Props) {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(Boolean(imgPath));
  const [caption, setCaption] = useState('');

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    setIsOpen(Boolean(imgPath));
  }, [imgPath]);

  useEffect(() => {
    if (!isOpen && imgPath) {
      const t = setTimeout(() => onChange(null), 300);
      return () => clearTimeout(t);
    }
  }, [isOpen, imgPath, onChange]);

  const onCancel = useCallback(() => setIsOpen(false), []);

  const viewable = useMemo(() => images.filter(path => !isAudio(path)), [images]);

  const currentIndex = useMemo(() => {
    if (!imgPath) return -1;
    return viewable.findIndex(path => path === imgPath);
  }, [imgPath, viewable]);

  const showPrev = useCallback(() => {
    if (currentIndex <= 0) return;
    onChange(viewable[currentIndex - 1]);
  }, [currentIndex, onChange, viewable]);

  const showNext = useCallback(() => {
    if (currentIndex === -1 || currentIndex >= viewable.length - 1) return;
    onChange(viewable[currentIndex + 1]);
  }, [currentIndex, onChange, viewable]);

  useEffect(() => {
    if (!imgPath || isAudio(imgPath)) {
      setCaption('');
      return;
    }
    let cancelled = false;
    setCaption('');
    apiClient
      .post('/api/caption/get', { imgPath })
      .then(res => res.data)
      .then(data => {
        if (cancelled) return;
        setCaption(data ? `${data}` : '');
      })
      .catch(() => {
        if (!cancelled) setCaption('');
      });
    return () => {
      cancelled = true;
    };
  }, [imgPath]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;
      switch (event.key) {
        case 'Escape':
          onCancel();
          break;
        case 'ArrowLeft':
          event.preventDefault();
          showPrev();
          break;
        case 'ArrowRight':
          event.preventDefault();
          showNext();
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel, showPrev, showNext]);

  if (!mounted) return null;

  const atStart = currentIndex <= 0;
  const atEnd = currentIndex === -1 || currentIndex >= viewable.length - 1;

  return createPortal(
    <Dialog open={isOpen} onClose={onCancel} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4 text-center">
          <DialogPanel
            transition
            className="relative transform rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in max-w-[95%] max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95 flex flex-col overflow-hidden"
          >
            <div className="relative overflow-hidden flex items-center justify-center">
              {imgPath &&
                !isAudio(imgPath) &&
                (isVideo(imgPath) ? (
                  <video
                    key={imgPath}
                    src={`/api/img/${encodeURIComponent(imgPath)}`}
                    className="w-auto h-auto max-w-[95vw] max-h-[calc(100vh-9rem)] object-contain"
                    preload="none"
                    playsInline
                    loop
                    autoPlay
                    controls
                  />
                ) : (
                  <img
                    key={imgPath}
                    src={`/api/img/${encodeURIComponent(imgPath)}`}
                    alt={fileName(imgPath)}
                    className="w-auto h-auto max-w-[95vw] max-h-[calc(100vh-9rem)] object-contain"
                  />
                ))}
              {viewable.length > 1 && (
                <>
                  <button
                    type="button"
                    aria-label="Previous image"
                    disabled={atStart}
                    className="absolute left-2 top-1/2 -translate-y-1/2 bg-gray-900/80 text-white rounded-full p-2 disabled:opacity-30"
                    onClick={showPrev}
                  >
                    <ChevronLeft />
                  </button>
                  <button
                    type="button"
                    aria-label="Next image"
                    disabled={atEnd}
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-gray-900/80 text-white rounded-full p-2 disabled:opacity-30"
                    onClick={showNext}
                  >
                    <ChevronRight />
                  </button>
                </>
              )}
            </div>
            <div className="shrink-0 bg-gray-950 text-sm px-4 py-2 max-h-32 overflow-auto">
              {imgPath && <div className="text-xs text-gray-400 truncate">{fileName(imgPath)}</div>}
              {caption && <div className="whitespace-pre-wrap break-words mt-1">{caption}</div>}
            </div>
            <button
              type="button"
              aria-label="Close"
              className="absolute top-2 right-2 bg-gray-900 rounded-full p-1 text-white opacity-50 hover:opacity-100"
              onClick={onCancel}
            >
              <X />
            </button>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
