/**
 * Render a SmallRef after we have all of the necessary data.
 */

import classNames from 'classnames';
import React from 'react';

import {IconName} from '../../../../Icon';
import {Tooltip} from '../../../../Tooltip';
import {Link} from '../pages/common/Links';
import {getErrorReason} from '../pages/wfReactInterface/utilities';
import {SmallRefIcon} from './SmallRefIcon';

type SmallRefLoadedProps = {
  icon: IconName;
  url: string;
  label?: string;
  error: Error | null;
  noLink?: boolean;
  suffix?: React.ReactNode;
};

export const SmallRefLoaded = ({
  icon,
  url,
  label,
  error,
  noLink = false,
  suffix,
}: SmallRefLoadedProps) => {
  const content = (
    <div
      className={classNames(
        'flex h-full items-center gap-4 font-semibold text-moon-700',
        {
          'cursor-pointer hover:text-teal-500': !error && !noLink,
          'line-through': error,
          'cursor-default': noLink,
        }
      )}>
      <SmallRefIcon icon={icon} />
      {label && (
        <div className="min-w-0 flex-1 overflow-hidden overflow-ellipsis whitespace-nowrap">
          {label}
        </div>
      )}
      {suffix && <div className="ml-2 flex-shrink-0">{suffix}</div>}
    </div>
  );
  if (error) {
    const reason = getErrorReason(error);
    return (
      <Tooltip
        trigger={<div className="w-full">{content}</div>}
        noTriggerWrap
        content={reason}
      />
    );
  }
  if (noLink) {
    return <div className="w-full cursor-default">{content}</div>;
  }
  return (
    <Link $variant="secondary" to={url} className="w-full">
      {content}
    </Link>
  );
};
