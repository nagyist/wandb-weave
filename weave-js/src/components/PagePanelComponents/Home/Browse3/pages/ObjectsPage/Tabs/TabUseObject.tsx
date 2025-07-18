import {Box} from '@mui/material';
import React from 'react';

import {isValidVarName} from '../../../../../../../core/util/var';
import {abbreviateRef} from '../../../../../../../util/refs';
import {CopyableText} from '../../../../../../CopyableText';
import {DocLink} from '../../common/Links';
import {TabUseBanner} from '../../common/TabUseBanner';

type TabUseObjectProps = {
  name: string;
  uri: string;
  entityName: string;
  projectName: string;
  versionIndex: number;
};

export const TabUseObject = ({
  name,
  uri,
  entityName,
  projectName,
  versionIndex,
}: TabUseObjectProps) => {
  const pythonName = isValidVarName(name) ? name : 'obj';
  const long = `weave.init('${entityName}/${projectName}')
${pythonName} = weave.ref('${name}:v${versionIndex}').get()`;

  return (
    <Box className="text-sm">
      <TabUseBanner>
        See{' '}
        <DocLink
          path="guides/tracking/objects#getting-an-object-back"
          text="Weave docs on refs"
        />{' '}
        for more information.
      </TabUseBanner>

      <Box mt={2}>
        The ref for this object version is:
        <CopyableText text={uri} />
      </Box>
      <Box mt={2}>
        Use the following code to retrieve this object version:
        <CopyableText
          language="python"
          text={`${pythonName} = weave.ref("${abbreviateRef(uri)}").get()`}
          copyText={`${pythonName} = weave.ref("${uri}").get()`}
          tooltipText="Click to copy unabridged string"
        />
        <div className="mt-8">or</div>
        <CopyableText language="python" text={long} />
      </Box>
    </Box>
  );
};
