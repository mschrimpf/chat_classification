package de.lab.hcm.textemotion;

/**
 * Created by chi-tai on 05.05.15.
 *
 * Chi-Tai Dang
 * HCM-Lab
 * University of Augsburg
 */

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.Log;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Pattern;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

@SuppressLint("NewApi")
public class TextEmotionActivity extends Activity {

    private static final String TAG = "TextEmotionActivity";
    private static final int NR_OF_CATEGORIES = 68;

    private TextView mEvalResultTextView = null;
    private ImageView mResultImageView = null;
    private EditText mTestInputEditText = null;
    private DataParser mDataParser;

    public SortedMap<String, List<Integer>> catMap = null;

    public String[] emotions = null;
    private Set<Integer> usedCategories;
    public Map<String, Attribute> attributesMap = null;
    public FastVector<Attribute> attributesList = null;
    private Classifier cl;
    private Instances trainingData;
    private boolean ldaCoefficientsEnabled = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text_emotion);
        mDataParser = new DataParser(this);

        mEvalResultTextView = (TextView) findViewById(R.id.evalResult);
        mResultImageView = (ImageView) findViewById(R.id.result);
        mTestInputEditText = (EditText) findViewById(R.id.testText);
        mTestInputEditText.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                if (actionId == EditorInfo.IME_ACTION_DONE || (event != null && (event.getKeyCode() == KeyEvent.KEYCODE_ENTER))) {
                    InputMethodManager in = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);

                    in.hideSoftInputFromWindow(v.getApplicationWindowToken(), 0);
                    v.clearFocus();
                    return true;

                }
                return false;
            }
        });
        mTestInputEditText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                testText();
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        mResultImageView.setBackground(getResources().getDrawable(R.drawable.neutral, null));

        // Load category map
        try {
            catMap = mDataParser.loadCatMap();
        } catch(Exception e) {
            throw new RuntimeException("unable to load categories map", e);
        }
        notRecognizedWords = new HashSet<>();
        //printTrainingSetAsCsv();
        //doBackwardsFeatureElimination();
        doForwardFeatureSelection();

        //hardcoded feature set
        //usedCategories = new HashSet<>();
        //Collections.addAll(usedCategories, 7, 16, 18, 19, 59, 66, 3, 13, 14, 15);
        //train();
        //count(notRecognizedWords);
    }

    //caches the matched categories for a string
    private static class PreprocessedString {
        String text;
        Set<Integer> matchingCategories;

        public PreprocessedString(String text, Set<Integer> matchingCategories) {
            this.text = text;
            this.matchingCategories = matchingCategories;
        }
    }

    private void doBackwardsFeatureElimination() {
        // Load training set
        List<List<PreprocessedString>> sampleSets = loadTrainingSet();
        //start with all features selected
        usedCategories = new HashSet<>();
        for(int i = 1; i <= NR_OF_CATEGORIES; i++) {
            usedCategories.add(i);
        }
        Log.w(TAG, "Used features: " + TextUtils.join(", ", usedCategories));
        double errorRate = train(sampleSets);
        //try to remove as many features as possible
        while(true) {
            //which feature to remove next?
            int bestCategoryToRemove = -1;
            //copying the list is necessary to avoid ConcurrentModificationException
            for(Integer candidate : new ArrayList<Integer>(usedCategories)) {
                Log.i(TAG, "candidate: " + candidate);
                usedCategories.remove(candidate);
                double newErrorRate = train(sampleSets);
                if(newErrorRate < errorRate) {
                    errorRate = newErrorRate;
                    bestCategoryToRemove = candidate;
                }
                //after testing; add it again
                usedCategories.add(candidate);
            }
            //no feature found?
            if(bestCategoryToRemove == -1) {
                break;
            }
            //remove the feature
            usedCategories.remove(bestCategoryToRemove);
            //debug output
            Log.w(TAG, "Removed feature " + bestCategoryToRemove);
            Log.w(TAG, "Used features: " + TextUtils.join(", ", usedCategories));
            Log.w(TAG, "Success rate: " + (1 - errorRate));
        }
        //train the model with the selected features
        train(sampleSets);
    }

    private void doForwardFeatureSelection() {
        // Load training set
        List<List<PreprocessedString>> sampleSets = loadTrainingSet();
        //start with no features selected
        usedCategories = new HashSet<>();
        Log.w(TAG, "Used features: " + TextUtils.join(", ", usedCategories));
        double errorRate = train(sampleSets);
        //try to remove as many features as possible
        while(true) {
            //which feature to remove next?
            int bestCategoryToAdd = -1;
            ArrayList<Integer> candidates = new ArrayList<>();
            for(int i = 1; i <= NR_OF_CATEGORIES; i++) {
                if(!usedCategories.contains(i)) candidates.add(i);
            }
            for(Integer candidate : candidates) {
                Log.i(TAG, "candidate: " + candidate);
                usedCategories.add(candidate);
                double newErrorRate = train(sampleSets);
                if(newErrorRate < errorRate) {
                    errorRate = newErrorRate;
                    bestCategoryToAdd = candidate;
                }
                //after testing; remove it again
                usedCategories.remove(candidate);
            }
            //no feature found?
            if(bestCategoryToAdd == -1) {
                break;
            }
            //remove the feature
            usedCategories.add(bestCategoryToAdd);
            //debug output
            Log.w(TAG, "Added feature " + bestCategoryToAdd);
            Log.w(TAG, "Used features: " + TextUtils.join(", ", usedCategories));
            Log.w(TAG, "Success rate: " + (1 - errorRate));
        }
        //train the model with the selected features
        train(sampleSets);
    }

    // Load the training set
    private List<List<PreprocessedString>> loadTrainingSet() {
        try {
            emotions = new String[]{"Positive", "Negative", "Neutral"};
            List<List<PreprocessedString>> sampleSets = new ArrayList<>();
            for(String basename : emotions) {
                List<String> strs = mDataParser.loadDataLines(basename + ".txt");
                List<PreprocessedString> currentSampleSet = new ArrayList<>(strs.size());
                for(String str : strs) {
                    str = str.trim();
                    currentSampleSet.add(new PreprocessedString(str, getMatchingCategories(str)));
                }
                sampleSets.add(currentSampleSet);
            }
            return sampleSets;
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    //makes the whole table exportable as CSV
    private void printTrainingSetAsCsv() {
        List<List<PreprocessedString>> data = loadTrainingSet();
        StringBuilder csvBuilder = new StringBuilder();
        for(int i = 0; i < data.size(); i++) {
            String result = emotions[i];
            List<PreprocessedString> samples = data.get(i);;
            for(PreprocessedString ppstr : samples) {
                csvBuilder.append(ppstr.text).append("\t").append(result);
                for(int cat = 1; cat <= NR_OF_CATEGORIES; cat++) {
                    csvBuilder.append("\t").append(ppstr.matchingCategories.contains(cat) ? "yes": "no");
                }
                csvBuilder.append("\n");
            }
        }
        Log.d(TAG, "CSV dump:\n" + csvBuilder.toString());
        try {
            File logFile = File.createTempFile("dump", ".csv");
            if (!logFile.exists()) {
                logFile.createNewFile();
            }
            // BufferedWriter for performance
            BufferedWriter buf = new BufferedWriter(new FileWriter(logFile));
            buf.append(csvBuilder.toString());
            buf.newLine();
            buf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private double train() {
        return train(loadTrainingSet());
    }

    Set<Object> notRecognizedWords;

    private double train(List<List<PreprocessedString>> sampleSets) {
        Log.d(TAG, "Training...");

        try {
            attributesList = new FastVector<>(2);
            FastVector<String> fvClassVal = new FastVector<>(2);
            fvClassVal.addElement(emotions[0]);
            fvClassVal.addElement(emotions[1]);
            fvClassVal.addElement(emotions[2]);
            attributesList.addElement(new Attribute("emotion", fvClassVal));
            attributesList.addElement(new Attribute("isQuestion"));
            attributesList.addElement(new Attribute("ld0"));
            attributesList.addElement(new Attribute("ld1"));
            for(int i : usedCategories) {
                attributesList.addElement(new Attribute("liwcCategory" + i));
            }

            //store them in a map for faster access later
            attributesMap = new HashMap<>();
            for(Attribute attr : attributesList) {
                attributesMap.put(attr.name(), attr);
            }

            // Create weka instances and name them Text2Emo
            int overallSize = 0;
            for(List<PreprocessedString> samples : sampleSets) {
                overallSize += samples.size();
            }
            trainingData = new Instances("Text2Emo", attributesList, overallSize);
            /// Our first attribute (0) declares the class
            trainingData.setClassIndex(0);
            //add all the training data
            for(int i = 0; i < sampleSets.size(); i++) {
                classifyMore(trainingData, sampleSets.get(i), emotions[i]);
            }

            // Backup weka instances
            File trainFile = File.createTempFile("Training", ".arff");
            FileWriter fw = new FileWriter(trainFile);

            fw.write(trainingData.toString());
            fw.flush();
            fw.close();

            cl = new NaiveBayes();

            // train out classifier
            cl.buildClassifier(trainingData);

            // Test the model
            Evaluation eva = new Evaluation(trainingData);

            // 10-fold cross validation
            eva.crossValidateModel(cl, trainingData, 10, new Random(10));

            if (mEvalResultTextView != null) {
                String evals = eva.toSummaryString();
                evals += "\r\n" + eva.toClassDetailsString();
                evals += "\r\n" + eva.toMatrixString();
                mEvalResultTextView.setText(evals);
            }
            Log.d(TAG, eva.toSummaryString());
            Log.d(TAG, eva.toClassDetailsString());
            Log.d(TAG, eva.toMatrixString());
            return eva.errorRate();
        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
    }


    private void testText() {
        String line = mTestInputEditText.getText().toString();
        Log.d(TAG, "Texting: " + line);

        try {
            double[] result = classifyString(line);

            // Output emotion
            Log.d(TAG, emotions[0] + ": " + result[0]);
            Log.d(TAG, emotions[1] + ": " + result[1]);
            Log.d(TAG, emotions[2] + ": " + result[2]);

            double max = 0;
            int sel = 0;
            for (int i = 0; i < 3; i++) {
                if (result[i] > max) {
                    max = result[i];
                    sel = i;
                }
            }
            Log.d(TAG, "Ergebnis: " + emotions[sel]);
            final int value = sel;
            mResultImageView.post(new Runnable() {
                @Override
                public void run() {

                    switch (value) {
                        case 0:
                            mResultImageView.setBackground(getResources().getDrawable(R.drawable.positive, null));
                            break;
                        case 1:
                            mResultImageView.setBackground(getResources().getDrawable(R.drawable.negative, null));
                            break;
                        case 2:
                            mResultImageView.setBackground(getResources().getDrawable(R.drawable.neutral, null));
                            break;
                    }
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    // Used to fill our dataset with instances and according features
    public void classifyMore(Instances ret, List<PreprocessedString> strings, String emotion) {
        for (PreprocessedString ppstring : strings) {
            ret.add(getWEKAInstance(ppstring.text, ppstring.matchingCategories, emotion));
        }
    }

    // Used to classify a new text using our trained classifier
    private double[] classifyString(String text) throws Exception {
        Instance iUse = getWEKAInstance(text, null);
        iUse.setDataset(trainingData);
        return cl.distributionForInstance(iUse);
    }

    public Instance getWEKAInstance(String text, String emotion) {
        text = text.trim();
        Set<Integer> matchingCategories = getMatchingCategories(text);
        return getWEKAInstance(text, matchingCategories, emotion);
    }

    public Instance getWEKAInstance(String text, Set<Integer> matchingCategories, String emotion) {
        // Assign the first attributes (the class)
        Instance ret = new DenseInstance(attributesList.size());
        if (emotion != null) {
            ret.setValue(attributesMap.get("emotion"), emotion);
        }

        ret.setValue(attributesMap.get("isQuestion"), text.matches(".*\\?\\W*$") ? 1 : 0);

        for(int category : usedCategories){
            Attribute attr = attributesMap.get("liwcCategory" + category);
            ret.setValue(attr, matchingCategories.contains(category) ? 1 : 0);
        }

        if(ldaCoefficientsEnabled) {
            //calculate linear discriminant factors
            double[] lds = new double[LinearDiscriminantFactors.factors[0].length];
            for (int cat = 0; cat < NR_OF_CATEGORIES; cat++) {
                for (int i = 0; i < lds.length; i++) {
                    int featureAs01 = (matchingCategories.contains(cat) ? 1 : 0);
                    lds[i] += LinearDiscriminantFactors.factors[cat][i] * featureAs01;
                }
            }
            for (int i = 0; i < lds.length; i++) {
                Attribute attr = attributesMap.get("ld" + i);
                ret.setValue(attr, lds[i]);
            }
        }

        return ret;
    }

    private Set<Integer> getMatchingCategories(String text) {
        Set<Integer> matchingCategories = new HashSet<>();

        String[] wordSplit = text.toLowerCase().split(" ");
        for (String word : wordSplit) {
            if (word == null || Objects.equals(word, "")) {
                continue;
            }
            if (catMap.containsKey(word)) {
                /* already inside of the map */
                matchingCategories.addAll(catMap.get(word));
            } else {
                /* Look for every word which starts with first two (or one, if only one character) and check those */
                String prefix = word.length() > 1 ? word.substring(0, 2) : word.substring(0, 1);
                boolean matched = false;
                for (Map.Entry<String, List<Integer>> entry : filterPrefix(catMap, prefix).entrySet()) {
                    Pattern pattern = Pattern.compile(entry.getKey());
                    if (pattern.matcher(word).matches()) {
                        matchingCategories.addAll(entry.getValue());
                        matched = true;
                    }
                }
                if (!matched) {
                    String[] escapedWords = word.split("\\W");
                    for (String escapedWord : escapedWords) {
                        if (escapedWord == null || Objects.equals(escapedWord, "")) {
                            continue;
                        }
                        String escapedPrefix = escapedWord.length() > 1 ? escapedWord.substring(0, 2) : escapedWord.substring(0, 1);
                        for (Map.Entry<String, List<Integer>> entry : filterPrefix(catMap, escapedPrefix).entrySet()) {
                            Pattern pattern = Pattern.compile(entry.getKey());
                            if (pattern.matcher(escapedWord).matches()) {
                                matchingCategories.addAll(entry.getValue());
                                matched = true;
                            }
                        }
                        if (!matched) {
                            notRecognizedWords.add(escapedWord);
                        }
                    }
                }
            }
        }
        return matchingCategories;
    }

    private void count(Collection<String> collection) {
        final Map<String, Integer> map = new TreeMap<>();
        for (String string : collection) {
            if (map.containsKey(string)) {
                map.put(string, map.get(string) + 1);
            } else {
                map.put(string, 1);
            }
        }
        SortedSet<Map.Entry<String, Integer>> sortedMap = entriesSortedByValues(map);
        for (Map.Entry<String, Integer> pair : sortedMap) {
            Log.i(TAG, pair.getKey() + ": " + pair.getValue());
        }
    }

    static <K, V extends Comparable<? super V>>
    SortedSet<Map.Entry<K, V>> entriesSortedByValues(Map<K, V> map) {
        SortedSet<Map.Entry<K, V>> sortedEntries = new TreeSet<>(
                new Comparator<Map.Entry<K, V>>() {
                    @Override
                    public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
                        int res = e1.getValue().compareTo(e2.getValue());
                        return res != 0 ? res : 1;
                    }
                }
        );
        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    /* http://stackoverflow.com/questions/6713239/partial-search-in-hashmap */
    public <V> SortedMap<String, V> filterPrefix(SortedMap<String, V> baseMap, String prefix) {
        if (prefix.length() > 0) {
            char nextLetter = prefix.charAt(prefix.length() - 1);
            nextLetter++;
            String end = prefix.substring(0, prefix.length() - 1) + nextLetter;
            return baseMap.subMap(prefix, end);
        }
        return baseMap;
    }
}
