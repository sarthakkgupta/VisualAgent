import { useState, useEffect, useRef } from 'react';

interface TypeWriterProps {
  phrases: string[];
  typingSpeed?: number;
  deletingSpeed?: number;
  pauseDuration?: number;
  onTextChange?: (text: string) => void;
}

const TypeWriter: React.FC<TypeWriterProps> = ({ 
  phrases, 
  typingSpeed = 80, 
  deletingSpeed = 50, 
  pauseDuration = 1500,
  onTextChange = () => {} 
}) => {
  const [currentText, setCurrentText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isWaiting, setIsWaiting] = useState(false);

  // Keep track of the current phrase being typed
  const currentPhraseIndex = useRef(0);

  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout>;

    if (isWaiting) {
      timeout = setTimeout(() => {
        setIsWaiting(false);
        setIsDeleting(true);
      }, pauseDuration);
      return () => clearTimeout(timeout);
    }

    const currentPhrase = phrases[currentPhraseIndex.current];
    
    if (isDeleting) {
      if (currentText === '') {
        setIsDeleting(false);
        currentPhraseIndex.current = (currentPhraseIndex.current + 1) % phrases.length;
      } else {
        timeout = setTimeout(() => {
          setCurrentText(prev => prev.slice(0, -1));
          setCurrentIndex(prev => prev - 1);
        }, deletingSpeed);
      }
    } else {
      if (currentIndex < currentPhrase.length) {
        timeout = setTimeout(() => {
          setCurrentText(prev => prev + currentPhrase[currentIndex]);
          setCurrentIndex(prev => prev + 1);
        }, typingSpeed);
      } else {
        setIsWaiting(true);
      }
    }

    onTextChange(currentText);

    return () => clearTimeout(timeout);
  }, [
    currentText, 
    currentIndex, 
    isDeleting, 
    isWaiting, 
    phrases, 
    typingSpeed, 
    deletingSpeed, 
    pauseDuration,
    onTextChange
  ]);

  return <>{currentText}</>;
};

export default TypeWriter;