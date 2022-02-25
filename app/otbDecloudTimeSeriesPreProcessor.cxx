/*=========================================================================

     Copyright (c) INRAE 2020-2022. All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "itkFixedArray.h"
#include "itkObjectFactory.h"
#include "otbWrapperApplicationFactory.h"

// Application engine
#include "otbStandardFilterWatcher.h"
#include "itkFixedArray.h"

// Stack
#include "otbTensorflowSource.h"
#include "otbTensorflowCommon.h"
#include <vector>
#include <algorithm>
#include <numeric>

// Functor
#include "otbFunctorImageFilter.h"

// Channels slices
#include "otbMultiChannelExtractROI.h"

namespace otb
{

namespace Wrapper
{

// Name of the environment variable for the number of outputs
const std::string ENV_VAR_NOUTPUTS = "DECLOUD_PREPROCESSING_NOUTPUTS";

// Structure to store one timestamp and one index
template <class TimestampType>
struct TimestampWithIndex
{
  TimestampType timestamp;
  unsigned int  index;
};

// Sorting modes
enum SortMode
{
  ASC, // Ascending order
  DES, // Descending order
  ABS  // Ascending order computed from the abs(t-tref)
};

/*
 * \class PixelFunction
 *
 * \brief Computes the output pixel from
 * - SAR pixels (stacked in channels)
 * - Optical pixels (stacked in channels)
 * - pairs (list of pair of SAR and Optical input images indices)
 * - Nodata (of SAR, and Optical)
 * - Number of channels (in SAR, and Optical)
 *
 * The function that computes the output pixel, outputs the stacked [SAR, Optical] pixels in channels
 *
 * \ingroup OTBDecloud
 */
template <class TPixel, class TPairList>
class PixelFunction
{
public:
  PixelFunction() {}
  ~PixelFunction() {}
  bool
  operator!=(const PixelFunction &) const
  {
    return false;
  }
  bool
  operator==(const PixelFunction & other) const
  {
    return !(*this != other);
  }

  // Returns the output pixel size
  std::size_t
  OutputSize(std::array<size_t, 2ul> Inputs) const
  {
    return m_NbOutputImages * (m_SARNbBands + m_OptNbBands);
  }

  // Set parameters
  void
  SetParameters(TPairList    pairs,
                unsigned int sarNbBands,
                unsigned int optNbBands,
                float        sarNdVal,
                float        optNdVal,
                unsigned int nbOutputImages)
  {
    m_Pairs = pairs;
    m_SARNbBands = sarNbBands;
    m_OptNbBands = optNbBands;
    m_SARNoDataValue = sarNdVal;
    m_OptNoDataValue = optNdVal;
    m_NbOutputImages = nbOutputImages;
  }

  /*
   * Get the pixel of n input image
   */
  TPixel
  GetPixel(const TPixel & inPix, unsigned int idx, unsigned int nbBands) const
  {
    TPixel pix;
    pix.SetSize(nbBands);

    unsigned int outBand = 0;
    unsigned int start = idx * nbBands;
    for (unsigned int i = start; i < start + nbBands; i++)
    {
      pix[outBand] = inPix[i];
      outBand++;
    }

    return pix;
  }

  /*
   * Tell if the pixel is full on no-data values
   */
  bool
  IsNoData(const TPixel & pix, const float noDataValue) const
  {
    for (unsigned int i = 0; i < pix.GetSize(); i++)
      if (pix[i] != noDataValue)
        return false;
    return true;
  }

  /*
   * Compute output pixel.
   * inSARPix: pixel of the stacked SAR images (N x m_SARNbBands)
   * inOptPix: pixel of the stacker optical images (M x m_OptNbBands)
   */
  inline TPixel
  operator()(const TPixel & inSARPix, const TPixel & inOptPix) const
  {

    // Prepare output pixel
    TPixel outPix;
    outPix.SetSize(m_NbOutputImages * (m_SARNbBands + m_OptNbBands));
    unsigned int outBand = 0, i = 0;
    for (unsigned int imgIdx = 0; imgIdx < m_NbOutputImages; imgIdx++)
    {
      for (i = 0; i < m_SARNbBands; i++)
      {
        outPix[outBand] = m_SARNoDataValue;
        outBand++;
      }
      for (i = 0; i < m_OptNbBands; i++)
      {
        outPix[outBand] = m_OptNoDataValue;
        outBand++;
      }
    }

    // Index of the current pair
    unsigned int currentNbOutput = 0;

    // Iterate through pairs
    outBand = 0;
    for (auto pair = m_Pairs.begin(); pair != m_Pairs.end() && currentNbOutput < m_NbOutputImages; ++pair)
    {
      unsigned int sarIdx = pair->first;  // Index of the SAR image
      unsigned int optIdx = pair->second; // Index of the optical image

      // Read pixels of both SAR and optical images
      TPixel sarPix = GetPixel(inSARPix, sarIdx, m_SARNbBands);
      TPixel optPix = GetPixel(inOptPix, optIdx, m_OptNbBands);

      // Return the output pixel (concatenate SAR and optical pixel) if both pixels are not no-data
      if (!IsNoData(sarPix, m_SARNoDataValue) && !IsNoData(optPix, m_OptNoDataValue))
      {
        for (i = 0; i < m_SARNbBands; i++)
        {
          outPix[outBand] = sarPix[i];
          outBand++;
        }
        for (i = 0; i < m_OptNbBands; i++)
        {
          outPix[outBand] = optPix[i];
          outBand++;
        }
        currentNbOutput++;
      }

    } // next pair

    // Return the output pixel
    return outPix;
  }


private:
  unsigned int m_NbOutputImages;
  TPairList    m_Pairs;
  unsigned int m_SARNbBands;
  unsigned int m_OptNbBands;
  float        m_SARNoDataValue;
  float        m_OptNoDataValue;
}; // PixelFunction


/**
 * The OTB Application, that does the work with the functor.
 */
class DecloudTimeSeriesPreProcessor : public Application
{
public:
  // Here is a few declaration of types

  /** Standard class typedefs. */
  typedef DecloudTimeSeriesPreProcessor Self;
  typedef Application                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);
  itkTypeMacro(CRGAPreProcessor, Application);

  /** Typedefs for image concatenation */
  typedef TensorflowSource<FloatVectorImageType> TFSourceType;

  /** Typedefs for various stuff */
  typedef float                                                     DeltaTimestampType;
  typedef float                                                     TimestampType;
  typedef std::vector<TimestampType>                                TimestampList;
  typedef std::pair<unsigned int, unsigned int>                     IndicesPair;
  typedef std::vector<IndicesPair>                                  IndicesPairList;
  typedef TimestampWithIndex<TimestampType>                         TimestampWithIndexType;
  typedef std::vector<TimestampWithIndexType>                       TimestampWithIndexList;
  typedef std::pair<TimestampWithIndexType, TimestampWithIndexType> CandidatePairType;
  typedef std::vector<CandidatePairType>                            CandidatePairListType;


  /** functor */
  typedef FloatVectorImageType::PixelType           PixelType;
  typedef PixelFunction<PixelType, IndicesPairList> FunctorType;
  typedef otb::FunctorImageFilter<FunctorType>      FilterType;

  /** slicing */
  typedef otb::MultiChannelExtractROI<PixelType::ValueType, PixelType::ValueType> ExtractorType;


  void
  DoUpdateParameters()
  {}

  void
  DoInit()
  {

    // Documentation
    SetName("DecloudTimeSeriesPreProcessor");
    SetDescription("This application prepares input time series of SAR/Optical sync pairs.");
    SetDocLongDescription("This application takes as inputs : 1 optical time series, 1 SAR time series, and their "
                          "respective timestamps lists. Change the " +
                          ENV_VAR_NOUTPUTS + " environment variable to select the number of output images.");
    SetDocLimitations("None");
    SetDocAuthors("Remi Cresson, Nicolas Narcon");

    // Input time series
    AddParameter(ParameterType_InputImageList, "ilsar", "Input SAR images list");
    AddParameter(ParameterType_InputImageList, "ilopt", "Input optical images list");

    // Input timestamps
    AddParameter(ParameterType_StringList, "timestampssar", "Input SAR images timestamps list");
    AddParameter(ParameterType_StringList, "timestampsopt", "Input optical images timestamps list");

    // Sorting behavior
    AddParameter(ParameterType_Choice, "sorting", "The way images pairs are sorted");
    AddChoice("sorting.asc", "Sort pairs in ascending chronological order");
    AddChoice("sorting.des", "Sort pairs in descending chronological order");
    AddChoice("sorting.abs", "Sort pairs in descending absolute gap wrt. reference timestamp");
    AddParameter(ParameterType_String, "sorting.abs.reftimestamp", "Reference timestamp");

    // SAR-optical gap
    AddParameter(ParameterType_Float, "maxgap", "maximum gap between SAR and optical images in seconds (!!!");
    SetDefaultParameterFloat("maxgap", 144.0 * 3600.0);

    // No data parameters
    AddParameter(ParameterType_Float, "nodatasar", "No data value for SAR images");
    SetDefaultParameterFloat("nodatasar", 0.0);
    AddParameter(ParameterType_Float, "nodataopt", "No data value for optical images");
    SetDefaultParameterFloat("nodataopt", -10000.0);

    // Output images
    m_Outputs = std::max(otb::tf::GetEnvironmentVariableAsInt(ENV_VAR_NOUTPUTS), 1);
    for (int i = 1; i <= m_Outputs; i++)
    {
      std::stringstream sarKey, optKey;
      sarKey << "outsar" << i;
      optKey << "outopt" << i;
      AddParameter(ParameterType_OutputImage, sarKey.str(), "output SAR image");
      AddParameter(ParameterType_OutputImage, optKey.str(), "output optical image");
    }

    SetMultiWriting(true);
  }

  // Converts a std::string to a TimestampType
  TimestampType
  Str2Timestamp(std::string str)
  {
    return std::stof(str);
  }

  // Return a vector of timestamps with indices
  TimestampWithIndexList
  GetTimestampsWithIndices(const std::string key)
  {
    // Get timestamp from ParameterStringList
    otbAppLogINFO("Get timestamps of key " << key);
    TimestampWithIndexList ts;
    unsigned int           count = 0;
    for (auto & str : GetParameterStringList(key))
    {
      TimestampWithIndexType newTimestampWithIndex;
      newTimestampWithIndex.index = count;
      newTimestampWithIndex.timestamp = Str2Timestamp(str);
      ts.push_back(newTimestampWithIndex);
      count++;
    }

    return ts;
  }

  // Sort the elements from the timestamp
  // The function modifies the "ts" vector.
  void
  SortTimestampsWithIndices(TimestampWithIndexList & ts)
  {
    // Sort timestamp (3 different strategies)
    SortMode sortMode = static_cast<SortMode>(GetParameterInt("sorting"));
    if (sortMode == ASC)
    {
      otbAppLogINFO("Sorting timestamps in ascending order");
      std::sort(ts.begin(), ts.end(), [](const TimestampWithIndexType & a, const TimestampWithIndexType & b) {
        return a.timestamp < b.timestamp;
      });
    }
    else if (sortMode == DES)
    {
      otbAppLogINFO("Sorting timestamps in descending order");
      std::sort(ts.begin(), ts.end(), [](const TimestampWithIndexType & a, const TimestampWithIndexType & b) {
        return a.timestamp > b.timestamp;
      });
    }
    else if (sortMode == ABS)
    {
      const TimestampType refTimestamp = Str2Timestamp(GetParameterAsString("sorting.abs.reftimestamp"));
      otbAppLogINFO("Sorting timestamps in ascending order from the gap with reference timestamp " << refTimestamp);
      std::sort(
        ts.begin(), ts.end(), [refTimestamp](const TimestampWithIndexType & a, const TimestampWithIndexType & b) {
          return std::abs(a.timestamp - refTimestamp) < std::abs(b.timestamp - refTimestamp);
        });
    }
    else
      otbAppLogCRITICAL("Wrong sorting mode");
  }

  // This function return a std::vector of pairs of SAR and Optical images indices.
  // The function uses the images timestamps to form the pairs.
  // After this function, the timestamps are not used anymore.
  IndicesPairList
  GetCandidatesPairs()
  {
    // Get images timestamps and indices
    TimestampWithIndexList sarTsWithIdxList = GetTimestampsWithIndices("timestampssar");
    TimestampWithIndexList optTsWithIdxList = GetTimestampsWithIndices("timestampsopt");

    // Get maxgap
    const DeltaTimestampType maxgap = GetParameterFloat("maxgap");
    if (maxgap < 3600.0)
      otbAppLogWARNING("maxgap is small (" << maxgap << " seconds). Did you miss to convert maxgap in seconds?");

    // Sort the optical images timestamps using ASC, DES or ABS strategy, depending on
    // the application parameter choice "sorting".
    SortTimestampsWithIndices(optTsWithIdxList);

    // Iterate over optical images, since they are freshly re-ordered
    IndicesPairList indicesPairs;
    for (auto optTsWithIdx : optTsWithIdxList)
    {
      // Timestamp of the optical image
      TimestampType optTs = optTsWithIdx.timestamp;

      // Index of the optical image, in the application parameter input image list
      unsigned int optIdx = optTsWithIdx.index;

      // For each optical image, we sort the available SAR images from the closest to the farthest
      // Indices are sorted such as the first index is the closest to the optical image timestamp
      std::sort(sarTsWithIdxList.begin(),
                sarTsWithIdxList.end(),
                [&](const TimestampWithIndexType i, const TimestampWithIndexType j) {
                  return std::abs(i.timestamp - optTs) < std::abs(j.timestamp - optTs);
                });

      // Now we pick all SAR images satisfying the maxgap, in the sorted order.
      // And we add them into the pairs list.
      for (auto sarTsWithIdx : sarTsWithIdxList)
        if (std::abs(sarTsWithIdx.timestamp - optTs) <= maxgap)
          indicesPairs.push_back({ sarTsWithIdx.index, optIdx });
    }

    otbAppLogINFO("Candidate pairs of indices:");
    for (auto & indicesPair : indicesPairs)
      otbAppLogINFO(<< "\t"
                    << "SAR: " << indicesPair.first << " ("
                    << std::to_string(sarTsWithIdxList[indicesPair.first].timestamp) << ") "
                    << "OPT: " << indicesPair.second << " ("
                    << std::to_string(optTsWithIdxList[indicesPair.second].timestamp) << ")");

    if (indicesPairs.size() == 0)
    {
      otbAppLogFATAL(<< "No S1/S2 pairs found. You could try to increase the maxgap and/or double check the dates of your timeseries");
    }
    return indicesPairs;
  }

  // This function prepares the input layerstack for each source, and populates the factorised list of pairs (outPairs)
  //  sarSrc: SAR source (modified in the function)
  //  optSrc: Optical source (modified in the function)
  //  inIndicesPairs: a std::vector of std::pairs of indices. It describes the original paired images with their
  //  indices. outIndicesPairs: a std::vector of std::pairs of indices (modified in the function). It describes the
  //  paired images with their indices, after we have removed all unused ones. inSARList: input SAR FloatVectorImageList
  //  inOptList: input optical FloatVectorImageList
  void
  InstantiateSources(TFSourceType &                    sarSrc,
                     TFSourceType &                    optSrc,
                     const IndicesPairList &           inIndicesPairs,
                     IndicesPairList &                 outIndicesPairs,
                     FloatVectorImageListType::Pointer inSARList,
                     FloatVectorImageListType::Pointer inOptList)
  {

    otbAppLogINFO("Preparing input images stacks");

    // New images list, that will contain only the images that are in "inPairs"
    FloatVectorImageListType::Pointer sarList = FloatVectorImageListType::New();
    FloatVectorImageListType::Pointer optList = FloatVectorImageListType::New();

    // Here we build SAR and optical stacks, and update pairs with the indices of the actual images used
    outIndicesPairs.clear();
    std::vector<unsigned int> sarOldIdx, optOldIdx;
    unsigned int              sarNewIdxCounter = 0, optNewIdxCounter = 0;
    unsigned int              sarNewIdx, optNewIdx;
    for (const auto pair : inIndicesPairs)
    {
      // Original indices
      const unsigned int sarIdx = pair.first;
      const unsigned int optIdx = pair.second;

      // Add the new index if its not already in the list of used images
      auto sarSearch = std::find(sarOldIdx.begin(), sarOldIdx.end(), sarIdx);
      if (sarSearch == sarOldIdx.end())
      {
        otbAppLogINFO("\tAdd SAR image #" << sarIdx);
        sarOldIdx.push_back(sarIdx);
        sarList->PushBack(inSARList->GetNthElement(sarIdx));
        sarNewIdx = sarNewIdxCounter;
        sarNewIdxCounter++;
      }
      // Retrieve position in new index if its already in the list of used images
      else
      {
        sarNewIdx = sarSearch - sarOldIdx.begin();
      }

      // Add the new index if its not already in the list of used images
      auto optSearch = std::find(optOldIdx.begin(), optOldIdx.end(), optIdx);
      if (optSearch == optOldIdx.end())
      {
        otbAppLogINFO("\tAdd optical image #" << optIdx);
        optOldIdx.push_back(optIdx);
        optList->PushBack(inOptList->GetNthElement(optIdx));
        optNewIdx = optNewIdxCounter;
        optNewIdxCounter++;
      }
      // Retrieve position in new index if its already in the list of used images
      else
      {
        optNewIdx = optSearch - optOldIdx.begin();
      }

      // Update pairs with new indices
      otbAppLogINFO("\tNew indices: SAR image #" << sarIdx << " --> " << sarNewIdx << ", Optical image #" << optIdx
                                                 << " --> " << optNewIdx);
      outIndicesPairs.push_back({ sarNewIdx, optNewIdx });
    }

    // Set source
    sarSrc.Set(sarList);
    optSrc.Set(optList);
  }

  /**
   * Simple check on size of images lists, and timestamps lists.
   * Also prints stuff.
   */
  void
  CheckNumbers(const std::string imgsKey, const std::string timestampKey)
  {
    // Check that images and timestamps sizes match
    unsigned int nImgs = GetParameterImageList(imgsKey)->Size();
    unsigned int nTimestamps = GetParameterStringList(timestampKey).size();
    if (nTimestamps != nImgs)
      otbAppLogFATAL("There is " << nImgs << " input images at input " << imgsKey << " but " << nTimestamps
                                 << " timestamps for " << timestampKey);

    // Summarize timestamps
    otbAppLogINFO("Timestamps for key " << timestampKey << ":");
    for (auto timestamp : GetParameterStringList(timestampKey))
      otbAppLogINFO("\t" << timestamp);
  }

  /**
   * Set-up the filter, and the slicers (because SAR and Optical images are stacked together)
   * which is the last part of the pipeline
   */
  void
  InitFilter(FilterType::Pointer &                 filter,
             std::vector<ExtractorType::Pointer> & sarSlicers,
             std::vector<ExtractorType::Pointer> & optSlicers,
             IndicesPairList &                     indicesPairs,
             TFSourceType &                        sarSrc,
             TFSourceType &                        optSrc)
  {

    // Clear slicers lists
    sarSlicers.clear();
    optSlicers.clear();

    // Get the number of bands in images
    GetParameterImageList("ilsar")->GetNthElement(0)->UpdateOutputInformation();
    GetParameterImageList("ilopt")->GetNthElement(0)->UpdateOutputInformation();
    unsigned int sarNbBands = GetParameterImageList("ilsar")->GetNthElement(0)->GetNumberOfComponentsPerPixel();
    unsigned int optNbBands = GetParameterImageList("ilopt")->GetNthElement(0)->GetNumberOfComponentsPerPixel();
    otbAppLogINFO("Number of bands found in SAR images: " << sarNbBands);
    otbAppLogINFO("Number of bands found in Optical images: " << optNbBands);

    // No-data values
    float sarNoData = GetParameterFloat("nodatasar");
    float optNoData = GetParameterFloat("nodataopt");

    // Initialize filter
    filter = FilterType::New();
    filter->GetModifiableFunctor().SetParameters(indicesPairs, sarNbBands, optNbBands, sarNoData, optNoData, m_Outputs);
    filter->SetInputs(sarSrc.Get(), optSrc.Get());

    // Initialize slicers
    FloatVectorImageListType::Pointer sarList = FloatVectorImageListType::New();
    FloatVectorImageListType::Pointer optList = FloatVectorImageListType::New();
    unsigned int                      start = 1;
    for (int i = 1; i <= m_Outputs; i++)
    {
      // SAR image
      ExtractorType::Pointer sarSlicer = ExtractorType::New();
      sarSlicer->SetFirstChannel(start);
      sarSlicer->SetLastChannel(start + sarNbBands - 1);
      sarSlicer->SetInput(filter->GetOutput());
      sarSlicer->UpdateOutputInformation();
      sarSlicers.push_back(sarSlicer);

      // Optical image
      ExtractorType::Pointer optSlicer = ExtractorType::New();
      optSlicer->SetFirstChannel(sarNbBands + start);
      optSlicer->SetLastChannel(start + sarNbBands + optNbBands - 1);
      optSlicer->SetInput(filter->GetOutput());
      optSlicer->UpdateOutputInformation();
      optSlicers.push_back(optSlicer);

      start += sarNbBands + optNbBands;
    }
  }

  void
  DoExecute()
  {
    // Check that timestamps lists have the same length as images lists
    CheckNumbers("ilsar", "timestampssar");
    CheckNumbers("ilopt", "timestampsopt");

    // Form the pairs of images indices.
    // In this step, optical images are sorted using the ASC, DES, or ABS strategy.
    // The optical images are first sorted regarding their timestamps.
    // Then, for each optical image, available SAR images satisfying the
    // "maxgap" criterion are kept and pairs are formed (Pairs are formed
    // using the order of SAR images: they are sorted using ABS strategy
    // relatively to the current optical image).
    IndicesPairList indicesPairs = GetCandidatesPairs();

    // Prepare images stacks of input images that will be used use, and find
    // the indices of corresponding (SAR, Optical) pairs
    InstantiateSources(m_SARStack,                      // Selected SAR images stack (modified)
                       m_OptStack,                      // Selected optical images stack (modified)
                       indicesPairs,                    // (SAR, Optical) indices pairs list
                       m_PairsIndices,                  // List of pairs of indices for selected images (modified)
                       GetParameterImageList("ilsar"),  // Input SAR images list
                       GetParameterImageList("ilopt")); // Input optical images list


    // Initialize the filter that computes the output SAR and optical time series
    InitFilter(m_Filter,       // The filter that "drills" the input time series (modified)
               m_OutSAR,       // The output list of SAR images (modified)
               m_OutOpt,       // The output list of optical images (modified)
               m_PairsIndices, // List of pairs of indices for selected images
               m_SARStack,     // Selected SAR images stack
               m_OptStack);    // Selected optical images stack

    // Set outputs
    for (int i = 1; i <= m_Outputs; i++)
    {
      std::stringstream sarKey, optKey;
      sarKey << "outsar" << i;
      optKey << "outopt" << i;
      SetParameterOutputImage(sarKey.str(), m_OutSAR[i - 1]->GetOutput());
      SetParameterOutputImage(optKey.str(), m_OutOpt[i - 1]->GetOutput());
    }
  }

private:
  int                                 m_Outputs;              // Number of outputs
  TFSourceType                        m_SARStack, m_OptStack; // Layerstacks for inputs
  FilterType::Pointer                 m_Filter;               // Time series "drilling" filter
  IndicesPairList                     m_PairsIndices;         // List of pairs of indices for inputs
  std::vector<ExtractorType::Pointer> m_OutSAR, m_OutOpt;     // Channels slicers for outputs

}; // end of class

} // namespace Wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT(otb::Wrapper::DecloudTimeSeriesPreProcessor)
