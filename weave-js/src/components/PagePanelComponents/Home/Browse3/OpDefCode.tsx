import Editor from '@monaco-editor/react';
import Box from '@mui/material/Box';
import {Loading} from '@wandb/weave/components/Loading';
import React, {FC} from 'react';

import {sanitizeString} from '../../../../util/sanitizeSecrets';
import {Alert} from '../../../Alert';
import {useWFHooks} from './pages/wfReactInterface/context';

function detectLanguage(uri: string, code: string) {
  // Simple language detection based on file extension or content
  if (uri.endsWith('.py')) {
    return 'python';
  }
  if (uri.endsWith('.js') || uri.endsWith('.ts')) {
    return 'javascript';
  }
  if (code.includes('def ') || code.includes('import ')) {
    return 'python';
  }
  if (code.includes('function ') || code.includes('const ')) {
    return 'javascript';
  }
  return 'plaintext';
}

export const OpDefCode: FC<{uri: string; maxRowsInView?: number}> = ({
  uri,
  maxRowsInView,
}) => {
  const {
    derived: {useCodeForOpRef},
  } = useWFHooks();
  const text = useCodeForOpRef(uri);
  if (text.loading) {
    return (
      <Box
        sx={{
          height: '38px',
          width: '100%',
        }}>
        <Loading centered size={25} />
      </Box>
    );
  }

  if (text.result == null) {
    return (
      <Box
        sx={{
          margin: '10px 16px 0 10px',
        }}>
        <Alert severity="warning">No code found for this operation</Alert>
      </Box>
    );
  }

  const sanitized = sanitizeString(text.result ?? '');
  const detectedLanguage = detectLanguage(uri, sanitized);

  const inner = (
    <Editor
      height={'100%'}
      defaultLanguage={detectedLanguage}
      loading={text.loading}
      value={sanitized}
      options={{
        readOnly: true,
        minimap: {enabled: false},
        scrollBeyondLastLine: false,
        padding: {top: 10, bottom: 10},
      }}
    />
  );
  if (maxRowsInView) {
    const totalLines = sanitized.split('\n').length ?? 0;
    const showLines = Math.min(totalLines, maxRowsInView);
    const lineHeight = 18;
    const padding = 20;
    const height = showLines * lineHeight + padding + 'px';
    return <Box sx={{height}}>{inner}</Box>;
  }
  return inner;
};
