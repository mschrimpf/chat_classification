package de.lab.hcm.textemotion;

/**
 * Created by chi-tai on 05.05.15.

 * Chi-Tai Dang
 * HCM-Lab
 * University of Augsburg
 */

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SortedMap;
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
    
    private TextView mEvalResultTextView = null;
    private ImageView mResultImageView = null;
    private EditText mTestInputEditText = null;
    private DataParser mDataParser;
    
    private List<Category> categories = null;
    public SortedMap<String, List<Integer>> catMap = null;

    // Category, Count
    public Map<Integer, Integer> resultMap = new HashMap<>();
    private int ATTRIB_COUNT = 2;
    private int ATTRIB_CLASS = 0;
    public String[] emotions = null;
    public FastVector<Attribute> attributes = null;
    private Classifier cl;
    private Instances trainingData;
    
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
        train();
    }

    private void train()
    {
        Log.d(TAG, "Training...");

        try {
            emotions = new String[] {"Positive", "Negative", "Neutral"};

            ATTRIB_COUNT = 0;

            FastVector<String> fvClassVal = new FastVector<>(2);

            fvClassVal.addElement(emotions[0]);
            fvClassVal.addElement(emotions[1]);
            fvClassVal.addElement(emotions[2]);

            Attribute emotionClasses = new Attribute("Emotion", fvClassVal);
            ATTRIB_COUNT++;
            Attribute myFeature = new Attribute("myFeature");
            ATTRIB_COUNT++;

            // Create vector for attributes
            attributes = new FastVector<>(ATTRIB_COUNT);
            attributes.addElement(emotionClasses);
            attributes.addElement(myFeature);
            ATTRIB_CLASS = 0;

            // Load category map
            catMap = mDataParser.loadCatMap();

            // Load categories
            categories = mDataParser.loadCategories();

            // Load training set
            List<String> neutrals = mDataParser.loadDataLines("Neutral.txt"),
                    positives = mDataParser.loadDataLines("Positive.txt"),
                    negatives = mDataParser.loadDataLines("Negative.txt");


            // Create weka instances and name them Text2Emo
            trainingData = new Instances("Text2Emo", attributes, positives.size());

            /// Our first attribute (0) declares the class
            trainingData.setClassIndex(ATTRIB_CLASS);

            classifyMore(trainingData, positives, "Positive");
            classifyMore(trainingData, neutrals, "Neutral");
            classifyMore(trainingData, negatives, "Negative");

            // Backup weka instances
            File trainFile = File.createTempFile("Training", ".arff");
            FileWriter fw = new FileWriter(trainFile);

            fw.write(trainingData.toString()); fw.flush(); fw.close();

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
                mEvalResultTextView.setText (evals);
            }

            Log.d(TAG, eva.toSummaryString());
            Log.d(TAG, eva.toClassDetailsString());
            Log.d(TAG, eva.toMatrixString());
        }
        catch (Exception e) {
            e.printStackTrace();
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
    public void classifyMore(Instances ret, List<String> strings, String emotion) {
        for(String string: strings) {
            ret.add(getWEKAInstance(string, emotion));
        }
    }

    // Used to classify a new text using our trained classifier
    private double[] classifyString(String text) throws Exception {
        Instance iUse = getWEKAInstance(text, null);
        iUse.setDataset(trainingData);
        return cl.distributionForInstance(iUse);
    }

    /*
     * This is the "hot spot" where feature extraction has to be done...
     */
    public Instance getWEKAInstance(String text, String emotion) {
        // Assign the first attributes (the class)
        Instance ret = new DenseInstance(ATTRIB_COUNT);
        if(emotion != null) {
            ret.setValue(attributes.elementAt(0), emotion);
        }

        //
        // Do some feature extraction and calculation
        // Make use of the dictionary and word categories
        //
        float exampleValue;

        /* Gets all matches */
        String[] textSplit = text.split(" ");
        List<Integer> matches = new ArrayList<>();

        for (String input : textSplit){
            if (catMap.containsKey(input)){
                /* already inside of the map */
                matches.addAll(catMap.get(input));
            }else{
                /* Look for every word which starts with first two (or one, if only one character) and check those */
                String prefix = input.length() > 1 ? input.substring(0, 2) : input.substring(0, 1);
                for (Map.Entry<String, List<Integer>> entry : filterPrefix(catMap, prefix).entrySet()){
                    Pattern pattern = Pattern.compile(entry.getKey());
                    if (pattern.matcher(input).find()){
                        matches.addAll(entry.getValue());
                    }
                }
            }
        }

        // e.g. we use the length of the string
        exampleValue = text.length();

        ret.setValue(attributes.elementAt(1), exampleValue);
        return ret;
    }

    /* http://stackoverflow.com/questions/6713239/partial-search-in-hashmap */
    public <V> SortedMap<String, V> filterPrefix(SortedMap<String,V> baseMap, String prefix) {
        if(prefix.length() > 0) {
            char nextLetter = prefix.charAt(prefix.length() -1);
            String end = prefix.substring(0, prefix.length()-1) + nextLetter;
            return baseMap.subMap(prefix, end);
        }
        return baseMap;
    }
}
