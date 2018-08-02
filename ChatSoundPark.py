{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1.0.0'"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import re\n",
    "tf.__version__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Load the data\n",
    "# I'm only going to use a small portion of the data so that it load quickly on Kaggle.\n",
    "southpark = pd.read_csv(\"All-seasons.csv\")\n",
    "#simpsons = pd.read_csv(\"simpsons.csv\") # can't load on Kaggle, but you can do it on your own pc."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Season</th>\n",
       "      <th>Episode</th>\n",
       "      <th>Character</th>\n",
       "      <th>Line</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>Stan</td>\n",
       "      <td>You guys, you guys! Chef is going away. \\n</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>Kyle</td>\n",
       "      <td>Going away? For how long?\\n</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>Stan</td>\n",
       "      <td>Forever.\\n</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>Chef</td>\n",
       "      <td>I'm sorry boys.\\n</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>Stan</td>\n",
       "      <td>Chef said he's been bored, so he joining a gro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Season Episode Character                                               Line\n",
       "0     10       1      Stan         You guys, you guys! Chef is going away. \\n\n",
       "1     10       1      Kyle                        Going away? For how long?\\n\n",
       "2     10       1      Stan                                         Forever.\\n\n",
       "3     10       1      Chef                                  I'm sorry boys.\\n\n",
       "4     10       1      Stan  Chef said he's been bored, so he joining a gro..."
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "southpark.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "South Park lines:\n",
      "Line # 1\n",
      "You guys, you guys! Chef is going away. \n",
      "\n",
      "Line # 2\n",
      "Going away? For how long?\n",
      "\n",
      "Line # 3\n",
      "Forever.\n",
      "\n",
      "Line # 4\n",
      "I'm sorry boys.\n",
      "\n",
      "Line # 5\n",
      "Chef said he's been bored, so he joining a group called the Super Adventure Club. \n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"South Park lines:\")\n",
    "for i in range(0,5):\n",
    "    print(\"Line #\",i+1)\n",
    "    print(southpark.Line[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def clean_text(text):\n",
    "    '''Clean text by removing unnecessary characters and altering the format of words.'''\n",
    "\n",
    "    text = text.lower()\n",
    "    \n",
    "    text = re.sub(r\"\\n\", \"\",  text)\n",
    "    text = re.sub(r\"[-()]\", \"\", text)\n",
    "    text = re.sub(r\"\\.\", \" .\", text)\n",
    "    text = re.sub(r\"\\!\", \" !\", text)\n",
    "    text = re.sub(r\"\\?\", \" ?\", text)\n",
    "    text = re.sub(r\"\\,\", \" ,\", text)\n",
    "    text = re.sub(r\"i'm\", \"i am\", text)\n",
    "    text = re.sub(r\"he's\", \"he is\", text)\n",
    "    text = re.sub(r\"she's\", \"she is\", text)\n",
    "    text = re.sub(r\"it's\", \"it is\", text)\n",
    "    text = re.sub(r\"that's\", \"that is\", text)\n",
    "    text = re.sub(r\"what's\", \"that is\", text)\n",
    "    text = re.sub(r\"\\'ll\", \" will\", text)\n",
    "    text = re.sub(r\"\\'re\", \" are\", text)\n",
    "    text = re.sub(r\"won't\", \"will not\", text)\n",
    "    text = re.sub(r\"can't\", \"cannot\", text)\n",
    "    text = re.sub(r\"n't\", \" not\", text)\n",
    "    text = re.sub(r\"n'\", \"ng\", text)\n",
    "    text = re.sub(r\"ohh\", \"oh\", text)\n",
    "    text = re.sub(r\"ohhh\", \"oh\", text)\n",
    "    text = re.sub(r\"ohhhh\", \"oh\", text)\n",
    "    text = re.sub(r\"ohhhhh\", \"oh\", text)\n",
    "    text = re.sub(r\"ohhhhhh\", \"oh\", text)\n",
    "    text = re.sub(r\"ahh\", \"ah\", text)\n",
    "    \n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Clean the scripts and add them to the same list.\n",
    "text = []\n",
    "\n",
    "for line in southpark.Line:\n",
    "    text.append(clean_text(line))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "you guys , you guys ! chef is going away . \n",
      "going away ? for how long ?\n",
      "forever .\n",
      "i am sorry boys .\n",
      "chef said he is been bored , so he joining a group called the super adventure club . \n",
      "wow !\n",
      "chef ? ? what kind of questions do you think adventuring around the world is gonna answer ? !\n",
      "that is the meaning of life ? why are we here ?\n",
      "i hope you are making the right choice .\n",
      "i am gonna miss him .  i am gonna miss chef and i . . .and i do not know how to tell him ! \n",
      "dude , how are we gonna go on ? chef was our fuh . . .fffriend . \n",
      "and we will all miss you , chef ,  but we know you must do what your heart tells you . .\n",
      "byebye !\n",
      "goodbye !\n",
      "so long !\n",
      "so long , chef !\n",
      "goodbye , chef !\n",
      "goodbye , chef ! have a great time with the super adventure club !\n",
      "goodbye !  . .\n",
      "draw two card , fatass .\n"
     ]
    }
   ],
   "source": [
    "# Take a look at some of the text to ensure that it has been cleaned well.\n",
    "limit = 0\n",
    "for i in range(limit,limit+20):\n",
    "    print(text[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Find the length of lines\n",
    "lengths = []\n",
    "for line in text:\n",
    "    lengths.append(len(line.split()))\n",
    "\n",
    "# Create a dataframe so that the values can be inspected\n",
    "lengths = pd.DataFrame(lengths, columns=['counts'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>counts</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>400.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>13.092500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>14.833206</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>9.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>157.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           counts\n",
       "count  400.000000\n",
       "mean    13.092500\n",
       "std     14.833206\n",
       "min      2.000000\n",
       "25%      5.000000\n",
       "50%      9.000000\n",
       "75%     17.000000\n",
       "max    157.000000"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lengths.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "19.0\n",
      "21.0\n",
      "24.0\n",
      "34.0\n",
      "71.01\n"
     ]
    }
   ],
   "source": [
    "print(np.percentile(lengths, 80))\n",
    "print(np.percentile(lengths, 85))\n",
    "print(np.percentile(lengths, 90))\n",
    "print(np.percentile(lengths, 95))\n",
    "print(np.percentile(lengths, 99))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Limit the text we will use to the shorter 95%.\n",
    "max_line_length = 30\n",
    "\n",
    "short_text = []\n",
    "for line in text:\n",
    "    if len(line.split()) <= max_line_length:\n",
    "        short_text.append(line)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Create a dictionary for the frequency of the vocabulary\n",
    "vocab = {}\n",
    "for line in short_text:\n",
    "    for word in line.split():\n",
    "        if word not in vocab:\n",
    "            vocab[word] = 1\n",
    "        else:\n",
    "            vocab[word] += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Limit the vocabulary to words used more than 3 times.\n",
    "threshold = 3\n",
    "count = 0\n",
    "for k,v in vocab.items():\n",
    "    if v >= threshold:\n",
    "        count += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Size of total vocab: 737\n",
      "Size of vocab we will use: 215\n"
     ]
    }
   ],
   "source": [
    "print(\"Size of total vocab:\", len(vocab))\n",
    "print(\"Size of vocab we will use:\", count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# In case we want to use a different vocabulary sizes for the source and target text, \n",
    "# we can set different threshold values.\n",
    "# Nonetheless, we will create dictionaries to provide a unique integer for each word.\n",
    "source_vocab_to_int = {}\n",
    "\n",
    "word_num = 0\n",
    "for k,v in vocab.items():\n",
    "    if v >= threshold:\n",
    "        source_vocab_to_int[k] = word_num\n",
    "        word_num += 1\n",
    "        \n",
    "target_vocab_to_int = {}\n",
    "\n",
    "word_num = 0\n",
    "for k,v in vocab.items():\n",
    "    if v >= threshold:\n",
    "        target_vocab_to_int[k] = word_num\n",
    "        word_num += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Add the unique tokens to the vocabulary dictionaries.\n",
    "codes = ['<PAD>','<EOS>','<UNK>','<GO>']\n",
    "\n",
    "for code in codes:\n",
    "    source_vocab_to_int[code] = len(source_vocab_to_int)+1\n",
    "    \n",
    "for code in codes:\n",
    "    target_vocab_to_int[code] = len(target_vocab_to_int)+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Create dictionaries to map the unique integers to their respective words.\n",
    "# i.e. an inverse dictionary for vocab_to_int.\n",
    "source_int_to_vocab = {v_i: v for v, v_i in source_vocab_to_int.items()}\n",
    "target_int_to_vocab = {v_i: v for v, v_i in target_vocab_to_int.items()}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "219\n",
      "219\n",
      "219\n",
      "219\n"
     ]
    }
   ],
   "source": [
    "# Check the length of the dictionaries.\n",
    "print(len(source_vocab_to_int))\n",
    "print(len(source_int_to_vocab))\n",
    "print(len(target_vocab_to_int))\n",
    "print(len(target_int_to_vocab))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Create the source and target texts.\n",
    "# The target text is the line following the source text.\n",
    "source_text = short_text[:-1]\n",
    "target_text = short_text[1:]\n",
    "\n",
    "for i in range(len(target_text)):\n",
    "    target_text[i] += ' <EOS>'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "375\n",
      "375\n"
     ]
    }
   ],
   "source": [
    "# Check if the source and target text lengths match.\n",
    "print(len(source_text))\n",
    "print(len(target_text))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Convert the text to integers. \n",
    "# Replace any words that are not in the respective vocabulary with <UNK> (unknown)\n",
    "source_int = []\n",
    "for line in source_text:\n",
    "    sentence = []\n",
    "    for word in line.split():\n",
    "        if word not in source_vocab_to_int:\n",
    "            sentence.append(source_vocab_to_int['<UNK>'])\n",
    "        else:\n",
    "            sentence.append(source_vocab_to_int[word])\n",
    "    source_int.append(sentence)\n",
    "    \n",
    "target_int = []\n",
    "for line in target_text:\n",
    "    sentence = []\n",
    "    for word in line.split():\n",
    "        if word not in target_vocab_to_int:\n",
    "            sentence.append(target_vocab_to_int['<UNK>'])\n",
    "        else:\n",
    "            sentence.append(target_vocab_to_int[word])\n",
    "    target_int.append(sentence)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "375\n",
      "375\n"
     ]
    }
   ],
   "source": [
    "# Check the lengths\n",
    "print(len(source_int))\n",
    "print(len(target_int))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def model_inputs():\n",
    "    '''Create palceholders for inputs to the model'''\n",
    "    input_data = tf.placeholder(tf.int32, [None, None], name='input')\n",
    "    targets = tf.placeholder(tf.int32, [None, None], name='targets')\n",
    "    lr = tf.placeholder(tf.float32, name='learning_rate')\n",
    "    keep_prob = tf.placeholder(tf.float32, name='keep_prob')\n",
    "\n",
    "    return input_data, targets, lr, keep_prob\n",
    "def process_encoding_input(target_data, vocab_to_int, batch_size):\n",
    "    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''\n",
    "    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])\n",
    "    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)\n",
    "\n",
    "    return dec_input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length, attn_length):\n",
    "    '''Create the encoding layer'''\n",
    "    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)\n",
    "    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)\n",
    "    cell = tf.contrib.rnn.AttentionCellWrapper(drop, attn_length, state_is_tuple = True)\n",
    "    enc_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)\n",
    "    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,\n",
    "                                                   cell_bw = enc_cell,\n",
    "                                                   sequence_length = sequence_length,\n",
    "                                                   inputs = rnn_inputs, \n",
    "                                                   dtype=tf.float32)\n",
    "\n",
    "    return enc_state\n",
    "def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,\n",
    "                         output_fn, keep_prob):\n",
    "    '''Decode the training data'''\n",
    "    train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)\n",
    "    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(\n",
    "        dec_cell, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)\n",
    "    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)\n",
    "    return output_fn(train_pred_drop)\n",
    "def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,\n",
    "                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):\n",
    "    '''Decode the prediction data'''\n",
    "    infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(\n",
    "        output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size)\n",
    "    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)\n",
    "    return infer_logits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,\n",
    "                   num_layers, vocab_to_int, keep_prob, attn_length):\n",
    "    '''Create the decoding cell and input the parameters for the training and inference decoding layers'''\n",
    "    \n",
    "    with tf.variable_scope(\"decoding\") as decoding_scope:\n",
    "        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)\n",
    "        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)\n",
    "        cell = tf.contrib.rnn.AttentionCellWrapper(drop, attn_length, state_is_tuple = True)\n",
    "        dec_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)\n",
    "        \n",
    "        weights = tf.truncated_normal_initializer(stddev = 0.1)\n",
    "        biases = tf.zeros_initializer()\n",
    "        output_fn = lambda x: tf.contrib.layers.fully_connected(x, \n",
    "                                                                vocab_size, \n",
    "                                                                None, \n",
    "                                                                scope=decoding_scope,\n",
    "                                                                weights_initializer = weights,\n",
    "                                                                biases_initializer = biases)\n",
    "\n",
    "        train_logits = decoding_layer_train(\n",
    "            encoder_state[0], dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)\n",
    "        decoding_scope.reuse_variables()\n",
    "        infer_logits = decoding_layer_infer(encoder_state[0], dec_cell, dec_embeddings, vocab_to_int['<GO>'],\n",
    "                                            vocab_to_int['<EOS>'], sequence_length, vocab_size,\n",
    "                                            decoding_scope, output_fn, keep_prob)\n",
    "\n",
    "    return train_logits, infer_logits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,\n",
    "                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, vocab_to_int, attn_length):\n",
    "    \n",
    "    '''Use the previous functions to create the training and inference logits'''\n",
    "    \n",
    "    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size+1, enc_embedding_size)\n",
    "    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length, attn_length)\n",
    "\n",
    "    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)\n",
    "    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size+1, dec_embedding_size], -1.0, 1.0))\n",
    "    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)\n",
    "\n",
    "    train_logits, infer_logits = decoding_layer(dec_embed_input, dec_embeddings, enc_state, target_vocab_size+1, \n",
    "                                                sequence_length, rnn_size, num_layers, vocab_to_int, keep_prob, \n",
    "                                                attn_length)\n",
    "    \n",
    "    return train_logits, infer_logits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Set the parameters\n",
    "epochs = 10000\n",
    "batch_size = 128\n",
    "rnn_size = 512\n",
    "num_layers = 2\n",
    "encoding_embedding_size = 512\n",
    "decoding_embedding_size = 512\n",
    "attn_length = 10\n",
    "learning_rate = 0.0005\n",
    "keep_probability = 0.8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_graph = tf.Graph()\n",
    "with train_graph.as_default():\n",
    "    \n",
    "    # Load the model inputs\n",
    "    input_data, targets, lr, keep_prob = model_inputs()\n",
    "    # Sequence length will be the max line length for each batch\n",
    "    sequence_length = tf.placeholder_with_default(max_line_length, None, name='sequence_length')\n",
    "    input_shape = tf.shape(input_data)\n",
    "    \n",
    "    # Create the logits from the model\n",
    "    train_logits, inference_logits = seq2seq_model(\n",
    "        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), \n",
    "        len(target_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, \n",
    "        target_vocab_to_int, attn_length)\n",
    "    \n",
    "    # Create a tensor to be used for making predictions.\n",
    "    tf.identity(inference_logits, 'logits')\n",
    "    with tf.name_scope(\"optimization\"):\n",
    "        # Loss function\n",
    "        cost = tf.contrib.seq2seq.sequence_loss(\n",
    "            train_logits,\n",
    "            targets,\n",
    "            tf.ones([input_shape[0], sequence_length]))\n",
    "\n",
    "        # Optimizer\n",
    "        optimizer = tf.train.AdamOptimizer(learning_rate)\n",
    "\n",
    "        # Gradient Clipping\n",
    "        gradients = optimizer.compute_gradients(cost)\n",
    "        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]\n",
    "        train_op = optimizer.apply_gradients(capped_gradients)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def pad_sentence_batch(sentence_batch, vocab_to_int):\n",
    "    \"\"\"Pad lines with <PAD> so each line of a batch has the same length\"\"\"\n",
    "    max_sentence = max([len(sentence) for sentence in sentence_batch])\n",
    "    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]\n",
    "def batch_data(source, target, batch_size):\n",
    "    \"\"\"Batch source and target together\"\"\"\n",
    "    for batch_i in range(0, len(source)//batch_size):\n",
    "        start_i = batch_i * batch_size\n",
    "        source_batch = source[start_i:start_i + batch_size]\n",
    "        target_batch = target[start_i:start_i + batch_size]\n",
    "        yield (np.array(pad_sentence_batch(source_batch, source_vocab_to_int)), \n",
    "               np.array(pad_sentence_batch(target_batch, target_vocab_to_int)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "338\n",
      "37\n"
     ]
    }
   ],
   "source": [
    "train_valid_split = int(len(source_int)*0.1)\n",
    "\n",
    "train_source = source_int[train_valid_split:]\n",
    "train_target = target_int[train_valid_split:]\n",
    "\n",
    "valid_source = source_int[:train_valid_split]\n",
    "valid_target = target_int[:train_valid_split]\n",
    "\n",
    "print(len(train_source))\n",
    "print(len(valid_source))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch   1/10 Batch    0/2 - Loss:  0.109, Seconds: 1220.10\n",
      "New Record!\n",
      "Epoch   2/10 Batch    0/2 - Loss:  0.159, Seconds: 1425.35\n",
      "New Record!\n",
      "Epoch   3/10 Batch    0/2 - Loss:  0.089, Seconds: 1343.92\n",
      "New Record!\n",
      "Epoch   4/10 Batch    0/2 - Loss:  0.085, Seconds: 1333.72\n",
      "New Record!\n",
      "Epoch   5/10 Batch    0/2 - Loss:  0.074, Seconds: 1364.70\n",
      "New Record!\n",
      "Epoch   6/10 Batch    0/2 - Loss:  0.072, Seconds: 1372.88\n",
      "New Record!\n",
      "Epoch   7/10 Batch    0/2 - Loss:  0.067, Seconds: 1302.10\n",
      "New Record!\n",
      "Epoch   8/10 Batch    0/2 - Loss:  0.065, Seconds: 1293.69\n",
      "New Record!\n",
      "Epoch   9/10 Batch    0/2 - Loss:  0.064, Seconds: 1252.40\n",
      "New Record!\n",
      "Epoch  10/10 Batch    0/2 - Loss:  0.062, Seconds: 1277.56\n",
      "New Record!\n"
     ]
    }
   ],
   "source": [
    "import time\n",
    "\n",
    "learning_rate_decay = 0.95\n",
    "display_step = 50\n",
    "stop_early = 0\n",
    "stop = 10000\n",
    "total_train_loss = 0\n",
    "summary_valid_loss = []\n",
    "\n",
    "\n",
    "checkpoint = \"./best_model.ckpt\" \n",
    "\n",
    "with tf.Session(graph=train_graph) as sess:\n",
    "    sess.run(tf.global_variables_initializer())\n",
    "\n",
    "    for epoch_i in range(1, epochs+1):\n",
    "        for batch_i, (source_batch, target_batch) in enumerate(\n",
    "                batch_data(train_source, train_target, batch_size)):\n",
    "            start_time = time.time()\n",
    "            _, loss = sess.run(\n",
    "                [train_op, cost],\n",
    "                {input_data: source_batch,\n",
    "                 targets: target_batch,\n",
    "                 lr: learning_rate,\n",
    "                 sequence_length: target_batch.shape[1],\n",
    "                 keep_prob: keep_probability})\n",
    "\n",
    "            total_train_loss += loss\n",
    "            end_time = time.time()\n",
    "            batch_time = end_time - start_time\n",
    "            \n",
    "            if batch_i % display_step == 0:\n",
    "                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'\n",
    "                      .format(epoch_i,\n",
    "                              epochs, \n",
    "                              batch_i, \n",
    "                              len(train_source) // batch_size, \n",
    "                              total_train_loss / display_step, \n",
    "                              batch_time*display_step))\n",
    "                print('New Record!') \n",
    "                saver = tf.train.Saver() \n",
    "                saver.save(sess, checkpoint)\n",
    "                total_train_loss = 0\n",
    "\n",
    "            if batch_i % 235 == 0 and batch_i > 0:\n",
    "                total_valid_loss = 0\n",
    "                start_time = time.time()\n",
    "                for batch_ii, (source_batch, target_batch) in \\\n",
    "                        enumerate(batch_data(valid_source, valid_target, batch_size)):\n",
    "                    valid_loss = sess.run(\n",
    "                    cost, {input_data: source_batch,\n",
    "                           targets: target_batch,\n",
    "                           lr: learning_rate,\n",
    "                           sequence_length: target_batch.shape[1],\n",
    "                           keep_prob: 1})\n",
    "                    total_valid_loss += valid_loss\n",
    "                end_time = time.time()\n",
    "                batch_time = end_time - start_time\n",
    "                avg_valid_loss = total_valid_loss / (len(valid_source) / batch_size)\n",
    "                print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))\n",
    "                \n",
    "                learning_rate *= learning_rate_decay\n",
    "                \n",
    "                summary_valid_loss.append(avg_valid_loss)\n",
    "                print('New Record!') \n",
    "                saver = tf.train.Saver() \n",
    "                saver.save(sess, checkpoint)\n",
    "                if avg_valid_loss <= min(summary_valid_loss):\n",
    "                    print('New Record!') \n",
    "                    stop_early = 0\n",
    "                    saver = tf.train.Saver() \n",
    "                    saver.save(sess, checkpoint)\n",
    "                \n",
    "                else:\n",
    "                    print(\"No Improvement.\")\n",
    "                    stop_early += 1\n",
    "                    if stop_early == stop:\n",
    "                        break\n",
    "        if stop_early == stop:\n",
    "            print(\"Stopping Training.\")\n",
    "            break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def sentence_to_seq(sentence, vocab_to_int):\n",
    "    '''Prepare the predicted sentence for the model'''\n",
    "    \n",
    "    sentence = clean_text(sentence)\n",
    "    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.split()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# This part of the project won't work on Kaggle since it requires loading checkpoints of the model\n",
    "\n",
    "# To create your own input sentence\n",
    "input_sentence = 'hey'\n",
    "\n",
    "# To use an sentence from the data\n",
    "random = np.random.choice(len(short_text))\n",
    "input_sentence = short_text[random]\n",
    "\n",
    "# Clean the input sentence before it is used in the model\n",
    "input_sentence = 'i am sorry boys'\n",
    "\n",
    "input_sentence = sentence_to_seq(input_sentence, source_vocab_to_int)\n",
    "\n",
    "\n",
    "loaded_graph = tf.Graph()\n",
    "with tf.Session(graph=loaded_graph) as sess:\n",
    "#    # Load the saved model\n",
    "    loader = tf.train.import_meta_graph(checkpoint + '.meta')\n",
    "    loader.restore(sess, checkpoint)\n",
    "    \n",
    "    # Load the tensors to be used as inputs\n",
    "    input_data = loaded_graph.get_tensor_by_name('input:0')\n",
    "    logits = loaded_graph.get_tensor_by_name('logits:0')\n",
    "    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')\n",
    "    \n",
    "    response_logits = sess.run(logits, {input_data: [input_sentence],keep_prob: 1.0})[0]\n",
    "\n",
    "print('Input')\n",
    "print('  Word Ids:      {}'.format([i for i in input_sentence]))\n",
    "print('  Input Words: {}'.format([source_int_to_vocab[i] for i in input_sentence]))\n",
    "\n",
    "print('\\nResponse')\n",
    "print('  Word Ids:      {}'.format([i for i in np.argmax(response_logits, 1)]))\n",
    "print('  Response Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(response_logits, 1)]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
