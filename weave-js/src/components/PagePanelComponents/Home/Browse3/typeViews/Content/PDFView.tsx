import 'yet-another-react-lightbox/plugins/thumbnails.css';
import 'yet-another-react-lightbox/styles.css';
import 'yet-another-react-lightbox/plugins/counter.css';

import {StyledTooltip} from '@wandb/weave/components/DraggablePopups';
import {WaveLoader} from '@wandb/weave/components/Loaders/WaveLoader';
import {LoadingDots} from '@wandb/weave/components/LoadingDots';
import {Tailwind, TailwindContents} from '@wandb/weave/components/Tailwind';
import type {PDFDocumentProxy} from 'pdfjs-dist';
import React, {useCallback, useEffect, useRef, useState} from 'react';
import Lightbox, {Slide} from 'yet-another-react-lightbox';
import Counter from 'yet-another-react-lightbox/plugins/counter';
import Download from 'yet-another-react-lightbox/plugins/download';
import Thumbnails from 'yet-another-react-lightbox/plugins/thumbnails';

import {useWFHooks} from '../../pages/wfReactInterface/context';
import {
  ContentMetadataTooltip,
  DownloadButton,
  getIconName,
  IconWithText,
  saveBlob,
} from './Shared';
import {ContentViewMetadataLoadedProps} from './types';

type PDFViewProps = {
  blob: Blob;
  open: boolean;
  onClose: () => void;
  onDownload?: () => void;
};

// This marker is used to satisfy the type checker while signifying that
// we should show a loading indicator.
const makePlaceholder = (page: number): Slide => ({
  type: 'image' as const,
  src: '',
  width: -1,
  height: page, // PDF pages are 1-indexed
});

export const PDFContent = (props: ContentViewMetadataLoadedProps) => {
  const [contentResult, setContentResult] = useState<Blob | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const {useFileContent} = useWFHooks();
  const {metadata, project, entity, content} = props;
  const {filename, size, mimetype} = metadata;

  const contentContent = useFileContent({
    entity,
    project,
    digest: content,
    skip: !isDownloading,
  });

  // Store the last non-null content content result in state
  // We do this because passing skip: true to useFileContent will
  // result in contentContent.result getting returned as null even
  // if it was previously downloaded successfully.
  useEffect(() => {
    if (contentContent.result) {
      const blob = new Blob([contentContent.result], {
        type: mimetype,
      });

      setContentResult(blob);
      setIsDownloading(false);
    }
  }, [contentContent.result, mimetype]);

  const doSave = useCallback(() => {
    if (!contentResult) {
      console.error('No content result');
      return;
    }
    saveBlob(contentResult, filename);
  }, [contentResult, filename]);

  const downloadContent = () => {
    if (!contentResult && !isDownloading) {
      setIsDownloading(true);
    } else if (contentResult) {
      // We really want to know if we are duplicating these large downloads
      console.warn('Attempted to download previously loaded content.');
    }
  };

  const openPreview = () => {
    setShowPreview(true);
    if (!contentResult && !isDownloading) {
      downloadContent();
    }
  };

  const closePreview = () => {
    setShowPreview(false);
  };
  const iconName = getIconName(mimetype);

  const iconWithText = (
    <div>
      <IconWithText
        iconName={iconName}
        filename={filename}
        onClick={openPreview}
      />
    </div>
  );

  const preview = showPreview && contentResult && (
    <PDFView
      open={true}
      onClose={closePreview}
      blob={contentResult}
      onDownload={doSave}
    />
  );

  if (showPreview) {
    return (
      <TailwindContents>
        {iconWithText}
        {preview}
      </TailwindContents>
    );
  }

  const tooltipTrigger = (
    <StyledTooltip
      enterDelay={500}
      title={
        <TailwindContents>
          <ContentMetadataTooltip
            filename={filename}
            mimetype={mimetype}
            size={size}
          />
          <div className="text-sm">
            <div className="mt-8 text-center text-xs">
              Click icon or filename to preview, button to download
            </div>
          </div>
        </TailwindContents>
      }>
      {iconWithText}
    </StyledTooltip>
  );

  return (
    <TailwindContents>
      <div className="group flex items-center gap-4">
        {tooltipTrigger}
        <div className="opacity-0 group-hover:opacity-100">
          <DownloadButton isDownloading={isDownloading} doSave={doSave} />
        </div>
      </div>
    </TailwindContents>
  );
};

export const PDFView = ({blob, open, onClose, onDownload}: PDFViewProps) => {
  // We need to maintain the page index ourselves because we are changing the
  // slides array dynamically (swapping in actual rendered pages for placeholders)
  // and if we let the lightbox manage its own state it would reset to the first page.
  const [index, setIndex] = useState(0);

  const [docProxy, setDocProxy] = useState<PDFDocumentProxy | null>(null);
  const [slides, setSlides] = useState<Slide[]>([makePlaceholder(-1)]);
  const renderTasksRef = useRef<Map<number, boolean>>(new Map());

  const plugins = docProxy
    ? slides.length > 1
      ? [Thumbnails, Counter]
      : [Counter]
    : [];
  if (onDownload) {
    plugins.push(Download);
  }
  const download = onDownload
    ? {
        download: onDownload,
      }
    : undefined;

  useEffect(() => {
    const loadPDF = async () => {
      try {
        // Dynamically import pdfjs-dist
        const pdfjsLib = await import('pdfjs-dist');
        // Set worker source dynamically
        pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
          'pdfjs-dist/build/pdf.worker.min.mjs',
          import.meta.url
        ).toString();

        // It may seem strange to pass in a Blob and then ask it for an ArrayBuffer,
        // when we get an ArrayBuffer back from our API layer, but this is preventing
        // a "Cannot perform Construct on a detached ArrayBuffer" error in pdfjs.
        const doc = await pdfjsLib.getDocument(await blob.arrayBuffer())
          .promise;
        setDocProxy(doc);
        setSlides(
          Array.from(
            {length: doc.numPages},
            (_, index) => makePlaceholder(index + 1) // PDF pages are 1-indexed
          )
        );
      } catch (error) {
        console.error('Error loading PDF:', error);
      }
    };

    loadPDF();
  }, [blob]);

  const renderPage = useCallback(
    async (pageIndex: number) => {
      // Check if we're already rendering this page
      if (renderTasksRef.current.get(pageIndex)) {
        return;
      }

      // Mark this page as being rendered
      renderTasksRef.current.set(pageIndex, true);

      try {
        if (!docProxy) {
          throw new Error('Document proxy is not set');
        }

        const page = await docProxy.getPage(pageIndex);
        const viewport = page.getViewport({scale: 2});
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        if (!context) {
          console.error('Failed to get canvas context');
          return;
        }

        canvas.height = viewport.height;
        canvas.width = viewport.width;
        await page.render({canvasContext: context, viewport}).promise;

        // Update the slides state with the rendered image
        setSlides(prevSlides => {
          const newSlides = [...prevSlides];
          newSlides[pageIndex - 1] = {
            type: 'image',
            src: canvas.toDataURL(),
          };
          return newSlides;
        });
      } catch (error) {
        console.error(`Error rendering page ${pageIndex}:`, error);
      } finally {
        // Mark this page as no longer being rendered
        renderTasksRef.current.set(pageIndex, false);
      }
    },
    [docProxy]
  );

  return (
    <Lightbox
      open={open}
      close={onClose}
      slides={slides}
      index={index}
      on={{
        view: ({index: currentIndex}) => setIndex(currentIndex),
      }}
      plugins={plugins}
      thumbnails={{
        position: 'start' as const,
        vignette: true,
      }}
      carousel={{finite: true}}
      download={download}
      counter={{container: {style: {top: 0}}}}
      render={{
        // Hide the prev/next buttons if there is only one page
        buttonPrev: slides.length <= 1 ? () => null : undefined,
        buttonNext: slides.length <= 1 ? () => null : undefined,

        thumbnail: ({slide}) => {
          const {src} = slide;
          if (src !== '') {
            // Image has been already been created for this slide, use default rendering.
            return null;
          }
          return <LoadingDots />;
        },
        slide: ({slide}) => {
          const {src, height: page} = slide;
          if (src !== '') {
            // Image has been already been created for this slide, use default rendering.
            return null;
          }

          if (page == null) {
            console.error('Unexpected - placeholder slide with no index');
            return null;
          }
          renderPage(page);

          // Show loading indicator while rendering
          return <WaveLoader size="huge" />;
        },
        slideFooter: () => {
          if (docProxy) {
            return null;
          }
          return (
            <Tailwind>
              <span className="fixed bottom-0 left-0 right-0 p-4 text-center text-white">
                Parsing document...
              </span>
            </Tailwind>
          );
        },
      }}
    />
  );
};
