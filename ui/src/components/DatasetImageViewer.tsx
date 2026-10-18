'use client';

import { useState, useEffect, useMemo, useCallback, useRef, KeyboardEvent as ReactKeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { ChevronLeft, ChevronRight, X } from 'lucide-react';
import classNames from 'classnames';
import { fetchCaption } from '@/utils/api';
import { isAudio, isVideo } from '@/utils/basic';
import AgentChatPanel from '@/components/AgentChatPanel';
import { useCaptionEdit } from '@/hooks/useCaptionEdit';
import { persistCaption } from '@/utils/persistCaption';

export interface CaptionSync {
  imgPath: string;
  caption: string;
}

interface Props {
  imgPath: string | null;
  images: string[];
  onChange: (nextPath: string | null) => void;
  onCaptionSaved?: (saved: CaptionSync) => void;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'TEXTAREA' || tag === 'INPUT';
}

function fileName(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

export default function DatasetImageViewer({ imgPath, images, onChange, onCaptionSaved }: Props) {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(Boolean(imgPath));
  const [caption, setCaption] = useState('');
  const [savedCaption, setSavedCaption] = useState('');
  const [isCaptionLoaded, setIsCaptionLoaded] = useState(false);
  const activePathRef = useRef<string | null>(imgPath);
  const captionEdit = useCaptionEdit({
    imgPath,
    caption,
    savedCaption,
    isCaptionLoaded,
    setCaption,
    setSavedCaption,
    onCaptionSaved,
  });

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
    activePathRef.current = imgPath;
    setCaption('');
    setSavedCaption('');
    setIsCaptionLoaded(false);
    if (!imgPath || isAudio(imgPath)) return;

    let cancelled = false;
    fetchCaption(imgPath)
      .then(text => {
        if (cancelled) return;
        setCaption(text);
        setSavedCaption(text.trim());
        setIsCaptionLoaded(true);
      })
      .catch(error => {
        if (!cancelled) console.error('Error fetching caption:', error);
      });
    return () => {
      cancelled = true;
    };
  }, [imgPath]);

  const saveCaption = () => {
    if (!imgPath || !isCaptionLoaded) return;
    const path = imgPath;
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) return;
    persistCaption(path, trimmedCaption)
      .then(() => {
        if (activePathRef.current === path) {
          setSavedCaption(trimmedCaption);
        }
        onCaptionSaved?.({ imgPath: path, caption: trimmedCaption });
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  const handleCaptionKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  useEffect(() => {
    // Headless UI closes the dialog from a window keydown listener. Stop
    // Escape during capture while the caption field is focused so the
    // caret stays put and the viewer stays open.
    const stopEscapeFromField = (event: KeyboardEvent) => {
      if (!isOpen || event.key !== 'Escape') return;
      if (!isEditableTarget(event.target)) return;
      event.preventDefault();
      event.stopPropagation();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;
      if (isEditableTarget(event.target)) return;
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
    window.addEventListener('keydown', stopEscapeFromField, true);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', stopEscapeFromField, true);
      window.removeEventListener('keydown', handleKeyDown);
    };
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
            className="relative transform rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in min-w-[75vw] max-w-[95%] max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95 flex flex-row items-stretch overflow-hidden"
          >
            <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
              <div className="relative flex items-center justify-center overflow-hidden">
                {imgPath &&
                  !isAudio(imgPath) &&
                  (isVideo(imgPath) ? (
                    <video
                      key={imgPath}
                      src={`/api/img/${encodeURIComponent(imgPath)}`}
                      className="w-auto h-auto max-w-full md:max-w-[calc(95vw-20rem)] max-h-[calc(100vh-13rem)] object-contain"
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
                      className="w-auto h-auto max-w-full md:max-w-[calc(95vw-20rem)] max-h-[calc(100vh-13rem)] object-contain"
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
              <div
                className={classNames('shrink-0 bg-gray-950 text-sm px-4 py-2 max-h-40 overflow-auto border-2', {
                  'border-blue-500': isCaptionLoaded && caption.trim() !== savedCaption,
                  'border-transparent': !isCaptionLoaded || caption.trim() === savedCaption,
                })}
              >
                {imgPath && <div className="text-xs text-gray-400 truncate">{fileName(imgPath)}</div>}
                {imgPath && isCaptionLoaded && (
                  <form
                    onSubmit={e => {
                      e.preventDefault();
                      saveCaption();
                    }}
                    onBlur={saveCaption}
                  >
                    <textarea
                      className="mt-1 w-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none"
                      value={caption}
                      rows={3}
                      onChange={e => setCaption(e.target.value)}
                      onKeyDown={handleCaptionKeyDown}
                    />
                  </form>
                )}
              </div>
              <button
                type="button"
                aria-label="Close"
                className="absolute top-2 right-2 bg-gray-900 rounded-full p-1 text-white opacity-50 hover:opacity-100"
                onClick={onCancel}
              >
                <X />
              </button>
            </div>
            <div className="relative hidden w-80 shrink-0 self-stretch md:block">
              <div className="absolute inset-0">
                <AgentChatPanel
                  key={imgPath ?? ''}
                  imgPath={imgPath}
                  caption={isCaptionLoaded ? caption : null}
                  onCaptionEdit={captionEdit.applyEdit}
                  captionUndo={captionEdit.undoAvailable ? { undo: captionEdit.undo } : null}
                />
              </div>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
